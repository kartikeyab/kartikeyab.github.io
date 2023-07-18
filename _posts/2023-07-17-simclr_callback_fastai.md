# Implementing SimCLR using a Callback in fast.ai 

Paper Link: (https://arxiv.org/abs/2002.05709)

## Overview
In this blog post, we'll implement SimCLR (A Simple Framework for Contrastive Learning of Visual Representations
) using fastai Callbacks.

## What are Callbacks?

Callbacks are a neat way to inject customizations in your training loop, without having to write the training loop again. 

The fast.ai book defines it nicely as: 

*"A callback is a piece of code that you write, and inject into another piece of code at some predefined point. In fact, callbacks have been used with deep learning training loops for years. The problem is that in previous libraries it was only possible to inject code in a small subset of places where this may have been required, and, more importantly, callbacks were not able to do all the things they needed to do."*

for example: say you want to multiply your training loss by 0.002 before backprop. With callbacks, you can go to the part of your training loop where the loss is calculated (*after_loss* in fast.ai context) and basically do: loss = 0.002 * loss.

sweet right?

![Callbacks](/images/callbacks.png)

## SimCLR summarised
SimCLR was a groundbreaking paper in the field of self-supervised learning in vision by Google. 

The idea behind self-supervised learning in vision is : make a model learn better visual representations without human supervision. SimCLR is based on Contrastive learning, where projections of augmented versions of the same image are brought closer (in the latent space), and poejections of augmented versions of other images are pushed apart, even if those images are of the same class of object. 

The trained model not only does well at identifying different transformations of the same image, but also learns representations of similar concepts (e.g., chairs vs. dogs), which later can be associated with labels through fine-tuning.

Here's an illustration of the SimCLR paper:

![SimCLR](/images/simclr.gif)


## Pseudocode
![Pcode](/images/pseudocode.png)