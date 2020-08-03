---
title: "Activation Functions: Let's see new activation functions"
last_modified_at: 2020-05-10T10:30:02-05:00
categories:
  - Blogs
tags:
  - Activation Function
excerpt: Activation functions are mathematical equations that determine the output of a neural network.
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

![Cover Page](https://missinglink.ai/wp-content/uploads/2018/11/activationfunction-1.png)

## GELU (Gaussian Error Linear Unit)

![Cover Page](/assets/images/gelu.png)

The GELU nonlinearity is the expected transformation of a stochastic regularizer which randomly applies the identity or zero map to a neuron’s input

#### 1. Equation:

![Cover Page](/assets/images/gelu_eq.png)

#### 2. GELU Experiments

Classfication Experiment: MNIST classification

![Cover Page](/assets/images/gelu_exp1.png)

Autoencoder Experiment: MNIST Autoencoder

![Cover Page](/assets/images/gelu_exp2.png)

Reference:

https://arxiv.org/pdf/1606.08415.pdf
https://github.com/hendrycks/GELUs

## LiSHT (Linearly Scaled Hyperbolic Tangent Activation)

![Cover Page](/assets/images/lisht.png)

#### 1. Equation:

![Cover Page](/assets/images/lisht_eq.png)

#### 2. LiSHT Experiments

Classification Experiment: MNIST & CIFAR10

![Cover Page](/assets/images/lisht_exp1.png)

Sentiment Classification Results using LSTM

![Cover Page](/assets/images/lisht_exp2.png)


Reference

https://arxiv.org/pdf/1901.05894.pdf

## SWISH

![Cover Page](/assets/images/swish.png)

#### 1. Equation:

![Cover Page](/assets/images/swish_eq.png)

#### 2. SWISH Experiments

Machine Translation

![Cover Page](/assets/images/swish_exp1.png)

Reference

https://arxiv.org/pdf/1710.05941.pdf

## Mish

![Cover Page](/assets/images/mish.png)

#### 1. Equation:

![Cover Page](/assets/images/mish_eq.png)

![Cover Page](/assets/images/mish_eq2.png)

#### 2. Mish Experiments

Output Landscape of a Random Neural Network

![Cover Page](/assets/images/mish_exp1.png)

Testing Accuracy v/s Number of Layers on MNIST

![Cover Page](/assets/images/mish_exp2.png)

Test Accuracy v/s Batch Size on CIFAR-10

![Cover Page](/assets/images/mish_exp3.png)


Reference

https://arxiv.org/pdf/1908.08681.pdf



##### Other Activation Functions

Rectified Activations: https://arxiv.org/pdf/1505.00853.pdf

Sparsemax: https://arxiv.org/pdf/1602.02068.pdf

<!-- hitwebcounter Code START -->
<a href="https://www.hitwebcounter.com" target="_blank">
<img src="https://hitwebcounter.com/counter/counter.php?page=7541383&style=0032&nbdigits=5&type=page&initCount=0" title="Web Counter" Alt="counter free"   border="0" >
</a>
