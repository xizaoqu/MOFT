"""This code was originally taken from https://github.com/google/prompt-to-
prompt and modified by Ferry."""
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union
from einops import rearrange
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

def noise_pred_regularization(noise_pred,
                              num_reg_steps=5,
                              num_ac_rolls=5,
                              kl_weight=20,
                              auto_coor_weight=20):
    e_t = noise_pred
    with torch.enable_grad():
        for _outer in range(num_reg_steps):
            if auto_coor_weight > 0:
                for _inner in range(num_ac_rolls):
                    _var = torch.autograd.Variable(e_t.detach().clone(),
                                                   requires_grad=True)
                    l_ac = auto_corr_loss(_var)
                    l_ac.backward()
                    _grad = _var.grad.detach() / num_ac_rolls
                    e_t = e_t - auto_coor_weight * _grad
            if kl_weight > 0:
                _var = torch.autograd.Variable(e_t.detach().clone(),
                                               requires_grad=True)
                l_kld = kl_divergence(_var)
                l_kld.backward()
                _grad = _var.grad.detach()
                e_t = e_t - kl_weight * _grad
            e_t = e_t.detach()
    noise_pred = e_t
    return noise_pred


class DDIMInversion:

    def __init__(self,
                 model,
                 num_reg_steps=0,
                 num_ac_rolls=0,
                 kl_weight=0,
                 auto_coor_weight=0,
                 scheduler = None,
                 cfg=False,
                 **kwargs):
        scheduler = scheduler
        self.model = model
        self.num_ddim_steps = kwargs.pop('num_ddim_steps', 25)
        self.guidance_scale = kwargs.pop('guidance_scale', 7.5)
        # self.tokenizer = self.model.tokenizer
        self.model.scheduler = scheduler
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None

        self.num_reg_steps = num_reg_steps
        self.num_ac_rolls = num_ac_rolls
        self.kl_weight = kl_weight
        self.auto_coor_weight = auto_coor_weight

        self.cfg = cfg

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - (self.scheduler.num_train_timesteps //
                                    self.scheduler.num_inference_steps)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (self.scheduler.alphas_cumprod[prev_timestep]
                             if prev_timestep >= 0 else
                             self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev)**0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample \
            + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray],
                  timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.num_train_timesteps //
            self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next)**0.5 * model_output
        next_sample = alpha_prod_t_next**0.5 * next_original_sample \
            + next_sample_direction
        return next_sample

    def forward_unet_features(self, z, t, encoder_hidden_states, layer_idx=[0], interp_res_h=256, interp_res_w=256):
        unet_output, all_intermediate_features = self.model.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True
            )

        all_return_features = []
        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = feat.permute(0,2,1,3,4)
            feat = feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4])
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return unet_output, all_return_features

    def get_noise_pred_single(self, latents, t, context, force_grad=False, return_intermediates=False):
        if return_intermediates:
            h = 64
            w = 64
            noise_pred, intermediate_feature = self.forward_unet_features(latents, t, context, layer_idx=[2], interp_res_h=h, interp_res_w=w)
            noise_pred = noise_pred.sample
            return noise_pred, intermediate_feature
        else:
            noise_pred = self.model.unet(latents,
                                        t,
                                        encoder_hidden_states=context)['sample']
            return noise_pred, None

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        # NOTE: we should set required grad as True for unet
        noise_pred = self.model.unet(latents_input,
                                     t,
                                     encoder_hidden_states=context)['sample']
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents, return_dict=False)[0]
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0,
                                      1).unsqueeze(0).to(self.model.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [''],
            padding='max_length',
            max_length=self.model.tokenizer.model_max_length,
            return_tensors='pt')
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding='max_length',
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, pred_step=None, return_intermediates=False):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        all_interfeature = []
        latent = latent.clone().detach()

        for i in tqdm(range(self.num_ddim_steps), desc='DDIM inversion'):
            
            if pred_step and i >= pred_step:
                break

            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1]
            # import pdb; pdb.set_trace()

            # if i == 0:
            #     noise_pred, intermediate_feature = self.get_noise_pred_single(latent, t, cond_embeddings, return_intermediates=True)
            #     torch.save(intermediate_feature[0], "temp_2/singtel_inv_"+str(i)+"_"+str(0)+'.pt')
            # else:

            if self.cfg:
                latent_model_input = torch.cat([latent] * 2)
                cond_embeddings =self.context.clone()
            else:
                latent_model_input = latent

            noise_pred, intermediate_feature = self.get_noise_pred_single(latent_model_input, t, cond_embeddings, return_intermediates=return_intermediates)
            noise_pred = noise_pred_regularization(noise_pred,
                                                   self.num_reg_steps,
                                                   self.num_ac_rolls,
                                                   self.kl_weight,
                                                   self.auto_coor_weight)

            if self.cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
            all_interfeature.append(intermediate_feature)

        return all_latent, all_interfeature

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image=None, latent=None, pred_step=None, return_intermediates=False):
        if latent is None:
            image = rearrange(image, "b c f h w -> (b f) c h w")
            latent = self.model.vae.encode(image)['latent_dist'].mean
            latent = latent * 0.18215
            latent = rearrange(latent, "(b f) c h w -> b c f h w", f=16)
        ddim_latents, inter_feat = self.ddim_loop(latent, pred_step=pred_step, return_intermediates=return_intermediates)
        return ddim_latents, inter_feat

    def run_inversion(self, image):
        return self.ddim_inversion(image)

    def run_reconstruct(self,
                        init_latent,
                        guidance_scale=None,
                        num_steps=None):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if guidance_scale is None:
            guidance_scale = 1
        else:
            guidance_scale = guidance_scale or self.guidance_scale
        print(f'Guidance Scale is {guidance_scale}')

        if num_steps is not None:
            orig_steps = self.num_ddim_steps
            self.model.scheduler.set_timesteps(num_steps)
            self.num_ddim_steps = num_steps
            print(f'Set num_steps to {num_steps}')

        latent = init_latent
        for i in tqdm(range(self.num_ddim_steps)):

            t = self.model.scheduler.timesteps[i]
            noise_pred_cond = self.get_noise_pred_single(
                latent, t, cond_embeddings)
            noise_pred_uncond = self.get_noise_pred_single(
                latent, t, uncond_embeddings)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond)

            latent = self.prev_step(noise_pred, t, latent)

        img = self.latent2image(latent)

        if num_steps is not None:
            self.model.scheduler.set_timesteps(orig_steps)
            self.num_ddim_steps = orig_steps

        return img

    def run_reconstruct_with_ddpm(self, init_latent, ddpm_steps):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)

        self.model.scheduler.set_timesteps(1000)

        latent = init_latent

        ddpm_steps_list = []
        ddim_steps_list = []
        for idx, t in enumerate(self.model.scheduler.timesteps):
            if idx == ddpm_steps - 1:
                break
            ddpm_steps_list.append(t)
            noise_pred = self.get_noise_pred_single(latent, t,
                                                    uncond_embeddings)
            latent = self.prev_step(noise_pred, t, latent)

        self.model.scheduler.set_timesteps(50)
        start_t = t
        for idx, t in enumerate(self.model.scheduler.timesteps):
            if t > start_t:
                continue

            ddim_steps_list.append(t)
            noise_pred = self.get_noise_pred_single(latent, t,
                                                    uncond_embeddings)
            latent = self.prev_step(noise_pred, t, latent)

        img = self.latent2image(latent)

        print(ddpm_steps_list)
        print(ddim_steps_list)
        return img
