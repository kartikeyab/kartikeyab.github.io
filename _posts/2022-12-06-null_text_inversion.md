# Null-text Inversion for Editing Real Images using Diffusion Models

Paper Link: (https://arxiv.org/abs/2211.09794)

## Overview

Text guided Diffusion models are all the rage these days. A lot of people are trying to figure out how one can manipulate or edit the images using these state of the art tools.

In this blogpost we explore a technique called Null-Text Inversion for real image editing using text input. The authors of the paper achieve this in three steps:

1. Inversion: The goal of this step is to find a noise vector that approximately produces the input image when fed with the input prompt into the diffusion process while preserving the editing capabilities of the model. Authors use DDIM inversion for this step.

| What goes in | What comes out |
|-|-|
| input image   | noise vector trajectory (pivots) |  
input prompt    |  

2. Null-Text Optimisation: Optmizing the unconditional text embedding to invert the input image and input prompt
inputs: noise vector(s) obtained in step 1, input prompt, unconditional embeddings
outputs: optimised unconditional embeddings

| What goes in | What comes out |
|-|-|
| noise vector(s) obtained in step 1  | optimised unconditional embeddings |   
input prompt   
unconditional embeddings | 

3. Edited Image generation: Diffuse the image with edited prompt, null-text embeddings and noise vector obtained from steps 1 and 2
inputs: edited text prompt, noise vector(s) obtained in step 1, optimised unconditional embeddings obtained in step 2
outputs: edited image

| What goes in | What comes out |
|-|-|
| edit text prompt  | edited image |   
noise vector(s) obtained in step 1   
optimised unconditional embeddings obtained in step 2 | 

## Code

### Install packages

```
!pip install diffusers torch torchvision transformers matplotlib fastai
```

### Imports
```
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from fastai.vision.all import *
from fastai.vision.all import *
from matplotlib import pyplot as plt

has_cuda = torch.cuda.is_available()
device = torch.device('cpu' if not has_cuda else 'cuda')
```

### Setup Pipe
```
model_id_or_path = "CompVis/stable-diffusion-v1-4"
scheduler = DDIMScheduler.from_config(model_id_or_path, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id_or_path,
    safety_checker=None,
    use_auth_token=True,
    scheduler = scheduler,
).to(device)

```

