# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

from matplotlib import pyplot as plt

import numpy as np
import torch
from tqdm import tqdm

import os.path as osp

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.my_unet import UNet3DConditionModel

from PIL import Image
from animatediff.models.ddim_inversion import DDIMInversion
import PIL
from torchvision import transforms

import torch.nn.functional as F
from functools import partial

import os
# from sklearn.preprocessing import StandardScaler

def get_tensor_from_video(video_path):
    """
    :param video_path: 视频文件地址
    :return: pytorch tensor
    """
    if not os.access(video_path, os.F_OK):
        print('测试文件不存在')
        return

    import cv2

    cap = cv2.VideoCapture(video_path)

    frames_list = []
    while(cap.isOpened()):
        ret,frame = cap.read()

        if not ret:
            break
        else:
            # 注意，opencv默认读取的为BGR通道组成模式，需要转换为RGB通道模式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
    cap.release()


    # 转换成tensor
    result_frames = torch.as_tensor(np.stack(frames_list))

    # 注意：此时result_frames组成的维度为[视频帧数量，宽，高，通道数]
    return result_frames

def get_curve(d_min, d_max, sig, change_frame):
    cuurent_frame = d_min.clone() * 0
    cuurent_frame[sig>0] = d_min[sig>0]
    cuurent_frame[sig<0] = d_max[sig<0]

    sig_flag = 1
    slope_list = []
    cf_idx = 1
    for i in range(0,15):
        if i>=change_frame[cf_idx]:
            cf_idx = cf_idx+1
            sig_flag *= -1
        interval = change_frame[cf_idx] - change_frame[cf_idx-1]
        slope_list.append((d_max-d_min)/interval*sig*sig_flag)

    frames_list = [cuurent_frame.clone()]
    for i in range(0,15):
        slope = slope_list[i]
        cuurent_frame = cuurent_frame + slope
        frames_list.append(cuurent_frame)
    frame_dim = torch.stack(frames_list,0)
    return frame_dim



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def draw_n_imgs(img_ten_list,
                sub_title_list,
                title,
                save_path,
                norm_across_imgs=True):
    n_imgs = len(img_ten_list)
    n_row = int(np.sqrt(n_imgs))
    n_col = int(np.ceil(n_imgs / n_row))

    # preprocess img_ten_list
    if norm_across_imgs:
        max_ = max([img_ten.max() for img_ten in img_ten_list]).numpy()

    img_list = []
    for img_ten in img_ten_list:
        b,c,f,w,h = img_ten.shape
        img_ten = img_ten.reshape(b,c,f*w,h)
        img = img_ten.permute(0, 2, 3, 1)[0].mean(-1).numpy()
        if norm_across_imgs:
            img = img / max_
        else:
            img = img / img.max()
        img = (img * 255).clip(0, 255)
        img_list.append(img)

    fig, axs = plt.subplots(n_row, n_col, figsize=(10, 5))
    fig.suptitle(title)
    # NOTE: dirty way to handle 1 row case
    if n_row == 1:
        axs = [axs]
    for i, (img, sub_title) in enumerate(zip(img_list, sub_title_list)):
        row = i // n_col
        col = i % n_col
        axs[row][col].imshow(img)
        axs[row][col].set_title(sub_title)
        axs[row][col].axis('off')

    fig.savefig(save_path)


def make_neighbour_points(point, r, h, w, factor, equ=False):

    X = torch.linspace(0, h, h)
    Y = torch.linspace(0, w, w)
    xx, yy = torch.meshgrid(X, Y)

    distance = ((xx - point[0])**2 + (yy - point[1])**2)**0.5

    if r == 0:
        relis, reljs = torch.nonzero(distance == distance.min(), as_tuple=True)
        return relis, reljs

    if equ:
        relis, reljs = torch.where(distance <= round(r / factor * h))
    else:
        relis, reljs = torch.where(distance < round(r / factor * h))

    return relis, reljs

def get_feat(feature, relis, reljs, h, w):
    gridh = relis / (h - 1) * 2 - 1
    gridw = reljs / (w - 1) * 2 - 1
    grid = torch.stack([gridw, gridh], dim=-1)[None, None, ...]
    target_feat = F.grid_sample(feature.float(),
                                grid.cuda(),
                                align_corners=True).squeeze(2)
    return target_feat

def cosine_guidance(feat1, feat2, scale=1, alpha=1, beta=1):

    bz, nch = feat1.shape[0], feat1.shape[1]

    feat1 = feat1.view(bz, nch, -1)
    feat2 = feat2.view(bz, nch, -1)
    if feat1.shape[2] > feat2.shape[2]:
        feat1 = feat1[:, :, :feat2.shape[2]]
    if feat2.shape[2] > feat1.shape[2]:
        feat2 = feat2[:, :, :feat1.shape[2]]

    guidance = (F.cosine_similarity(feat1, feat2) + 1) / 2

    return scale / (guidance.mean() * beta + alpha)

@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class AnimationMotionPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        ddim_inversion = False,
        copy_latents = False,
    ):
        super().__init__()
        self.l1loss = torch.nn.L1Loss()
        self.guidance_loss = partial(cosine_guidance,
                                                alpha=0.001,
                                                beta=1)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.ddim_inversion = ddim_inversion
        self.copy_latents = copy_latents
        self.scheduler = scheduler
        self.mseloss = torch.nn.MSELoss()


    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt): # check negative meaning
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def run_ddim_inversion(self, image, prompt, scheduler, pred_step=None, return_intermediates=False):
        inversioner = DDIMInversion(self,
                                    inversion_reg_steps = 5,
                                    inversion_ac_rolls = 5,
                                    inversion_kl_weight = 20,
                                    inversion_auto_coor_weight = 20,
                                    scheduler=scheduler,
                                    cfg=False)
        inversioner.init_prompt(prompt)
        latents, inter_feat = inversioner.ddim_inversion(image, pred_step=pred_step, return_intermediates=return_intermediates)

        encoder_hidden_state = inversioner.context
        _, cond = torch.chunk(encoder_hidden_state, 2)
        self.cond_embedding = cond
        # embedding = self.model.compute_text_embeddings(prompt)
        return latents, inter_feat
    
    def preprocess_image(self, image: Union[str, PIL.Image.Image,
                                            torch.Tensor]):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        if 512 not in image.size:
            preproc_pipeline = transforms.Compose([
                transforms.CenterCrop(min(image.size)),
                transforms.Resize(512),
            ])
            image = preproc_pipeline(image)

        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image[None, ...].permute(0, 3, 1, 2).cuda()

        # return preproc_pipeline(image)
        return image

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, prompt=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if self.ddim_inversion:
                video_tensor = get_tensor_from_video("__assets__/makevideo5_updown/10.gif")
                video = video_tensor.permute(3,0,1,2)[None].repeat(1,1,1,1,1).cuda()
                video = video.float() / 127.5 - 1
                latents = self.run_ddim_inversion(video, "photo of sky, mountains, forest", self.scheduler)
                latents = latents[-2]

            elif isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def save_inter_feat(self, video_path, prompts, outpath):

        video_tensor = get_tensor_from_video(video_path)
        video = video_tensor.permute(3,0,1,2)[None].repeat(1,1,1,1,1).cuda()
        video = video.float() / 127.5 - 1
        latents, inter_feat = self.run_ddim_inversion(video, prompts, self.scheduler, return_intermediates=True)
        inter_feat = inter_feat[5][0]
        torch.save(inter_feat.cpu(), outpath)

    def forward_unet_features(self, z, t, encoder_hidden_states, layer_idx=[0], interp_res_h=256, interp_res_w=256, shared_kv=False, prompt=False):
        unet_output, all_intermediate_features = self.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True,
            shared_kv=shared_kv
            )
        
        all_return_features = []
        for idx in layer_idx:
            feat = all_intermediate_features[idx]
            feat = feat.permute(0,2,1,3,4)
            feat = feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2], feat.shape[3], feat.shape[4])
            feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
            all_return_features.append(feat)
        return unet_output, all_return_features
    
    # @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        mask_gen=None,
        motion_guide=False,
        motion_scale=10,
        motion_guidance_weight=10,
        guidance_layer=[2],
        guidance_step=[3,12],
        reference_moft_path=None,
        dim_mask_path=None,
        mask_path=None,
        **kwargs,
    ):
        if isinstance(guidance_layer, str):
            guidance_layer = eval(guidance_layer)

        if mask_path is not None:
            mask_gen_list = []
            for i in range(len(mask_path)):
                mask_image = Image.open(mask_path[i])
                mask_gen = np.array(mask_image).astype(bool)
                mask_gen = torch.from_numpy(mask_gen).cuda().float()
                mask_gen = mask_gen.float()
                # import pdb;pdb.set_trace()
                mask_gen = F.interpolate(mask_gen[None,None], (64, 64), mode="bilinear")[0,0]
                mask_gen = mask_gen>0.5
                mask_gen = mask_gen.reshape(-1)
                mask_gen_list.append(mask_gen)

        if isinstance(motion_scale,str):
            motion_scale = eval(motion_scale)

        if isinstance(guidance_step,str):
            guidance_step = eval(guidance_step)

        with torch.no_grad():
            # Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, callback_steps)

            # Define call parameters
            # batch_size = 1 if isinstance(prompt, str) else len(prompt)
            batch_size = 1
            if latents is not None:
                batch_size = latents.shape[0]
            if isinstance(prompt, list):
                batch_size = len(prompt)

            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # do_classifier_free_guidance = False # hack

            # Encode input prompt
            prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
            if negative_prompt is not None:
                negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
            text_embeddings = self._encode_prompt(
                prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
            )

            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # Prepare latent variables
            num_channels_latents = self.unet.in_channels
            if latents is None:
                latents = self.prepare_latents(
                    batch_size * num_videos_per_prompt,
                    num_channels_latents,
                    video_length,
                    height,
                    width,
                    text_embeddings.dtype,
                    device,
                    generator,
                    latents,
                    prompt
                )
            latents = latents.to(device)

            use_parallel_latents = True
            if use_parallel_latents:
                parallel_latents = latents.clone()

            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)


        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                h = 64
                w = 64
                with torch.no_grad():
                    if use_parallel_latents:
                        parallel_latent_model_input = torch.cat([parallel_latents] * 2) if do_classifier_free_guidance else latents
                        parallel_latent_model_input = self.scheduler.scale_model_input(parallel_latent_model_input, t)

                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            with torch.no_grad():
                                parallel_noise_pred, _ = self.forward_unet_features(parallel_latent_model_input, t, text_embeddings, layer_idx=guidance_layer, interp_res_h=h, interp_res_w=w, shared_kv="replace")
                                parallel_noise_pred = parallel_noise_pred.sample

                if motion_guide and mask_gen is not None:
                    if i>=guidance_step[0] and i < guidance_step[1]:
                        if isinstance(motion_guidance_weight,str):
                            _motion_guidance_weight = eval(motion_guidance_weight)
                        else:
                            _motion_guidance_weight = motion_guidance_weight
                        
                        for repeat in range(2):
                            loss_total = 0.0

                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                latents.requires_grad_(True)
                                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                                gradient=None

                                noise_pred, intermediate_feature = self.forward_unet_features(latent_model_input, t, text_embeddings, layer_idx=[2], interp_res_h=h, interp_res_w=w, shared_kv="share")
                                noise_pred = noise_pred.sample

                                for guide_idx, guide_l  in enumerate(guidance_layer):
                                    if guide_idx > 0:
                                        break
                                    if do_classifier_free_guidance:
                                        inter_feature_origin = intermediate_feature[guide_idx].clone()
                                        inter_feature_origin = rearrange(inter_feature_origin, "(b f) c h w-> b f c h w", f=16)

                                        inter_feature = rearrange(inter_feature_origin, "b f c h w-> b f c (h w)")
                                        moft = inter_feature - inter_feature.mean(1)[:,None]

                                    else:
                                        raise NotImplementedError

                                    mask_g_total = mask_gen_list[0] & False
                                    for mi, mask_g in enumerate(mask_gen_list):
                                        moft_m = moft[:,:,:,mask_g]
                                        mask_g_total = mask_g_total | mask_g

                                        dim_mask = torch.load(dim_mask_path[mi])

                                        projected_data = moft_m[1].mean(-1)

                                        demo_flow = torch.load(reference_moft_path[mi]).to(projected_data.device)
                                        demo_flow = demo_flow[:,:,24:40,24:40].reshape(16,1280,-1).mean(-1)
                                        demo_flow = demo_flow - demo_flow.mean(0)[None]

                                        projected_data = projected_data[:,dim_mask]
                                        demo_flow = demo_flow[:,dim_mask]

                                        target = demo_flow
                                        loss_total += self.mseloss(projected_data, target.detach()) * _motion_guidance_weight

                            if loss_total > 0:
                                torch.autograd.backward(loss_total)
                                gradient = latents._grad
                                a,b,c,d,e = gradient.shape
                                gradient = gradient.reshape(a,b,c,d*e)
                                for ff in range(16):
                                    gradient[:,:,ff,~(mask_g_total)] = 0
                                gradient = gradient.reshape(a,b,c,d,e)
                                latents.requires_grad_(False)
                                latents = latents - gradient * 0.5
                            else:
                                gradient = None

                            latents.requires_grad_(False)

                    with torch.no_grad():
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        gradient=None
                        with torch.no_grad():
                            h = 64
                            w = 64
                            noise_pred, intermediate_feature = self.forward_unet_features(latent_model_input, t, text_embeddings, layer_idx=[2], interp_res_h=h, interp_res_w=w, prompt=prompt)
                            noise_pred = noise_pred.sample
                else:
                    with torch.no_grad():
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        gradient=None
                        with torch.no_grad():
                            h = 64
                            w = 64
                            noise_pred, intermediate_feature = self.forward_unet_features(latent_model_input, t, text_embeddings, layer_idx=[2], interp_res_h=h, interp_res_w=w, prompt=prompt)
                            noise_pred = noise_pred.sample

                # vis noise
                vis_noise = False
                if vis_noise:
                    pseduo_img = noise_pred[0,:3].permute(1,2,3,0)
                    B, H, W, C = pseduo_img.shape
                    out_img =  Image.fromarray(((pseduo_img.permute(1,0,2,3).reshape(H, B*W, C)+ 1) / 2 * 255).clip(
                            0, 255).cpu().numpy().astype(np.uint8))
                    out_img.save('attention_vis/noist_step_'+str(t.data.item())+'.png')         

                with torch.no_grad():
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                        if use_parallel_latents:
                            parallel_noise_pred_uncond, parallel_noise_pred_text = parallel_noise_pred.chunk(2)
                            parallel_noise_pred = parallel_noise_pred_uncond + guidance_scale * (parallel_noise_pred_text - parallel_noise_pred_uncond)

                    else:
                        noise_pred = noise_pred

                with torch.no_grad():
                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dict = self.scheduler.step(noise_pred, t, latents, gradient=gradient, **extra_step_kwargs)
                    latents = latents_dict['prev_sample']

                    if use_parallel_latents:
                        # compute the previous noisy sample x_t -> x_t-1
                        parallel_latents_dict = self.scheduler.step(parallel_noise_pred, t, parallel_latents, gradient=None, **extra_step_kwargs)
                        parallel_latents = parallel_latents_dict['prev_sample']

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        with torch.no_grad():
            video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
