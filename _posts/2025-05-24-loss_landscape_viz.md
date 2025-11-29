# Visualising Loss Landscape with Residual Connections

The concept of residual connections is pretty staightforward. Vanishing gradients is just one of the problems that residual connection solve. 
To truly appreciate it's beauty, we need to visualise their affects on gradients, activations, loss-landscape of the neural network.  

## What is a loss-landscape? How do you visualise it?
Loss surface (landscape) is a high-dimensional function that maps a model's parameters (weights and biases) to the 
scalar value of it's loss (MSE, cross-entropy, etc). 

Mathematically:
Loss Surface = L(θ)  
where θ ∈ ℝ^n are the parameters of the model

## Setup
We'll initialise two tiny models: a PlainNet (ConvNet without residual connections) and a ResNet (ConvNet with residual connections) and train on MNIST. In both cases, we will 
compare how the residual connections affect:
1. Loss Landscapes
2. Final Layer activations
3. Hessian Eigenvalue Spectrum (more on this later)

## Loss Landscape

To-do:
- a callback in fastai that plots loss surfaces? 
- 
