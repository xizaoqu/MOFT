import argparse
from omegaconf import OmegaConf
from diffusers import AutoencoderKL

from animatediff.utils.scheduler import AsyrpScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.my_unet import UNet3DConditionModel
from animatediff.pipelines.pipeline import AnimationMotionPipeline
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

import gc
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import imageio

import cv2
import os

from einops import rearrange
import torchvision

def load_pipeline():
    config = "configs/prompts/v1/v1-1-ToonYou.yaml"
    inference_config = "configs/inference/inference-v1.yaml"
    pretrained_model_path = "models/stable-diffusion-v1-5"

    model_config = OmegaConf.load(config)[0]
    motion_module = model_config.motion_module
    inference_config = OmegaConf.load(model_config.get(
        "inference_config", inference_config))

    ### >>> create validation pipeline >>> ###
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        assert False

    pipeline = AnimationMotionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=AsyrpScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule='linear',
                                    clip_sample=False,
                                    set_alpha_to_one=False),
    ).to("cuda")

    pipeline = load_weights(
        pipeline,
        motion_module_path=motion_module,
        motion_module_lora_configs=model_config.get(
            "motion_module_lora_configs", []),
        dreambooth_model_path=model_config.get("dreambooth_path", ""),
        lora_model_path=model_config.get("lora_model_path", ""),
        lora_alpha=model_config.get("lora_alpha", 0.8),
    ).to("cuda")

    return pipeline

def do_inversion(pipeline, video_path_list, outpath_list):
    for video_path, outpath in zip(video_path_list, outpath_list):
        pipeline.save_inter_feat(video_path=video_path,
                                    prompts='', outpath=outpath)
        
def convert_mp4_to_png(input_file, frame_idx):
    # Create the output folder if it doesn't exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(input_file)

    frames = []
    while True:
        ret, frame = cap.read()

        # Break the loop if we reached the end of the video
        if not ret:
            break

        frame = cv2.resize(frame, [400,400])
        # Convert the frame to NumPy array
        frame_array = np.asarray(frame)

        # Append the frame array to the list
        frames.append(frame_array)

    # Convert the list of frames to a NumPy array
    video_array = np.stack(frames)
    # Release the video capture object
    cap.release()

    return Image.fromarray(cv2.cvtColor(video_array[frame_idx],cv2.COLOR_BGR2RGB))

class Demo: # Borrowed from DIFT (https://diffusionfeatures.github.io/)

    def __init__(self, imgs, ft, img_size):
        self.ft = ft # NCHW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70, x=371, y=356, output_path=None):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)
        cos = nn.CosineSimilarity(dim=1)

        with torch.no_grad():
            src_ft = self.ft[0].unsqueeze(0)
            src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
            src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

            del src_ft
            gc.collect()
            torch.cuda.empty_cache()

            trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:])
            cos_map = cos(src_vec, trg_ft).cpu().numpy()  # N, H, W

            print(cos_map.max())
            print(cos_map.mean())

            del trg_ft
            gc.collect()
            torch.cuda.empty_cache()

            axes[0].clear()
            axes[0].imshow(self.imgs[0])
            axes[0].axis('off')
            axes[0].scatter(x, y, c='r', s=scatter_size)
            axes[0].set_title('source image')

            for i in range(1, self.num_imgs):
                max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                print(max_yx)
                axes[i].clear()

                heatmap = cos_map[i-1]
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                # heatmap[heatmap<0.5] =0

                # import pdb;pdb.set_trace()
                axes[i].imshow(self.imgs[i])
                axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                axes[i].axis('off')
                # axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=scatter_size)
                axes[i].set_title('target image')

            del cos_map
            del heatmap
            gc.collect()

        plt.savefig(output_path)
        plt.show()

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)