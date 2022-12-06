# Null-text Inversion for Editing Real Images using Diffusion Models

Paper Link: (https://arxiv.org/abs/2211.09794)

## Overview

Text guided Diffusion models are all the rage these days. A lot of people are trying to figure out how one can manipulate or edit the images using these state of the art tools.

In this blogpost we explore a technique called Null-Text Inversion for real image editing using text input. The authors of the paper achieve this in three steps:

1. Inversion: The goal of this step is to find a noise vector that approximately produces the input image when fed with the input prompt into the diffusion process while preserving the editing capabilities of the model. Authors use DDIM inversion for this step.

2. Null-Text Optimisation: Optmizing the unconditional text embedding to invert the input image and input prompt
inputs: noise vector(s) obtained in step 1, input prompt, unconditional embeddings
outputs: optimised unconditional embeddings

3. Edited Image generation: Diffuse the image with edited prompt, null-text embeddings and noise vector obtained from steps 1 and 2
inputs: edited text prompt, noise vector(s) obtained in step 1, optimised unconditional embeddings obtained in step 2
outputs: edited image

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

### Load inputs
```
image = Image.open("horse.png").convert("RGB")
image = image.resize((512, 512))

def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image

pp_image = preprocess_image(image)

generator = torch.Generator(device=device)
latent = pipe.vae.encode(pp_image.to(device)).latent_dist.sample(generator=generator) *  0.18215

input_prompt = "a horse running on a beach"
text_embeddings = pipe._encode_prompt(input_prompt, device, 1, False, None)

```

### DDIM Inversion
The goal of this step is to obtain a noise vector trajectory by gradually adding noise to the input image. Authors refer to this trajectory as a Pivot. Authors use a DDIM sampler to add noise in the input image over 50 timesteps.

![DDIM Inversion step](images/DDIM_inversion.jpg)

```
@torch.no_grad()
def ddim_inversion(latents, encoder_hidden_states, noise_scheduler, unet):
    next_latents = latents
    all_latents = [latents.detach().cpu()]

    # since we are adding noise to the image, we reverse the timesteps list to start at t=0
    reverse_timestep_list = reversed(noise_scheduler.timesteps)  
    
    for i in range(len(reverse_timestep_list)-1):
        timestep = reverse_timestep_list[i]
        next_timestep = reverse_timestep_list[i+1] 
        latent_model_input = noise_scheduler.scale_model_input(next_latents, timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states).sample

        alpha_prod_t =  noise_scheduler.alphas_cumprod[timestep]
        alpha_prod_t_next = noise_scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        f = (next_latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        next_latents = alpha_prod_t_next ** 0.5 * f + beta_prod_t_next ** 0.5 * noise_pred
        all_latents.append(next_latents.detach().cpu())


    return all_latents

num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps, device=device)
all_latents = ddim_inversion(latent, text_embeddings, pipe.scheduler, pipe.unet)
```

### Null-Text Optimisation
Unconditional or null-text embeddings significantly contribute to what the denoised image looks like. Our aim here is to find an optimised null-text emebdding that can help us "invert" our input image.

To do that, we use the pivots obtained in step-1 as labels (since every pivot holds information about the input image). We now initialise a null-text embedding, and create a denoising trajectory starting with the last latent (z50) obtained from Step-1. We calculate the mse-loss between the noise prediction obtained by using ( zt (t=50), input prompt and null-text embedding ) and the pivot zt-1. This loss is backpropagated and is used to update the null-text embeddings. The updated null-text emebddings will try to reduce the loss by predicting noise that is more similar to zt-1, thereby including more artificats of the input image, which in-turn helps in inverting it.

![DDIM Inversion step](images/null_text_opt.jpg)
