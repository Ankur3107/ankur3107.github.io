---
title: "Huggingface transformers on Macbook Pro M1 GPU"
last_modified_at: 2022-11-01T21:30:02-05:00
categories:
  - Blogs
tags:
  - mac-m1-gpu
  - huggingface
  - transformers
excerpt: Huggingface transformers on Macbook Pro M1 GPU
toc: true
toc_label: "Contents"
toc_icon: "cog"
---

# Introduction

When Apple has introduced ARM M1 series with unified GPU, I was very excited to use GPU for trying DL stuffs. I usually use Colab and Kaggle for my general training and exploration. Now this is right time to use M1 GPU as huggingface has also introduced mps device support ([mac m1 mps integration](https://github.com/huggingface/transformers/pull/18598)).
This enables users to leverage Apple M1 GPUs via mps device type in PyTorch for faster training and inference than CPU.

With M1 Macbook pro 2020 8-core GPU, I was able to get 2-3x improvement in the training time, compare to M1 CPU training on the same device. 


![Training Pic](https://raw.githubusercontent.com/Ankur3107/transformers-on-macbook-m1-gpu/main/training_pic.png)

<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2118670497450280"
     crossorigin="anonymous"></script>

# Install Pytorch on Macbook M1 GPU

##  Step 1: Install Xcode
    $ xcode-select --install

##  Step 2: Setup a new conda environment
    $ conda create -n torch-gpu python=3.8
    $ conda activate torch-gpu

##  Step 3: Install Pytorch
    $ conda install pytorch torchvision torchaudio -c pytorch-nightly

    # If not working with conda you may try pip
    $ pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

## Step 4: Sanity Check

```python
import torch
import math
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
```

# Hugging Face transformers Installation

## Step 1: Install Rust

    $ curl — proto ‘=https’ — tlsv1.2 -sSf https://sh.rustup.rs | sh

## Step 2: Install transformers

    $ pip install transformers


# Lets try to train QA model

    $ git clone https://github.com/Ankur3107/transformers-on-macbook-m1-gpu.git

    $ cd transformers-on-macbook-m1-gpu

    $ sh run.sh

# Benchmark

[sebastianraschka](https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html) has written a beautiful blog and benchmarked different M1 based systems on training and inferencing VGG16. Please do check.

![VGG16 Training on M1 device](https://sebastianraschka.com/images/blog/2022/pytorch-m1-gpu/vgg-benchmark-training.png)

![VGG16 Inferencing on M1 device](https://sebastianraschka.com/images/blog/2022/pytorch-m1-gpu/vgg-benchmark-inference.png)


# Reference

1. https://github.com/Ankur3107/transformers-on-macbook-m1-gpu
2. https://pytorch.org
3. https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
4. https://jamescalam.medium.com/hugging-face-and-sentence-transformers-on-m1-macs-4b12e40c21ce
5. https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/
6. https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html