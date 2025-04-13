
<br>
<p align="center">
<h1 align="center"><strong>Video Diffusion Models are Training-free Motion Interpreter and Controller</strong></h1>
  <p align="center">
    Zeqi Xiao&emsp;
    Yifan Zhou&emsp;
    Shuai Yang&emsp;
    Xingang Pan&emsp;
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2405.14864" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2405.14864-blue?">
  </a>
  <a href="https://xizaoqu.github.io/moft/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-&#x1F680-blue">
  </a>
</p>


## Installation

Install the environments by

```
conda create moft python==3.8
conda activate moft
pip install -r requirements.txt
```

Downloads checkpoints from [Animatediff](https://github.com/guoyww/AnimateDiff), [LoRA](https://huggingface.co/ckpt/realistic-vision-v20/blob/main/realisticVisionV20_v20.safetensors), and [SD-1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
Put them into the following structures:
```
models/
├── DreamBooth_LoRA
│   ├── realisticVisionV20_v20.safetensors
├── Motion_Module
│   ├── mm_sd_v15_v2.ckpt
├── stable-diffusion-v1-5
```


Run process.ipynb


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{
xiao2024video,
title={Video Diffusion Models are Training-free Motion Interpreter and Controller},
author={Zeqi Xiao and Yifan Zhou and Shuai Yang and Xingang Pan},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=ZvQ4Bn75kN}
}
```
