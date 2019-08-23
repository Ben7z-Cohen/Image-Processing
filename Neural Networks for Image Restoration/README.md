#Overview

This exercise deals with neural networks and their application to image restoration. In this exercise you
will develop a general workflow for training networks to restore corrupted images, and then apply this
workflow on two different tasks: (i) image denoising, (ii) image deblurring.

#Background

The method you will implement consists of the following three steps:
1. Collect “clean” images, apply simulated random corruptions, and extract small patches.
2. Train a neural network to map from corrupted patches to clean patches.
3. Given a corrupted image, use the trained network to restore the complete image by restoring each
patch separately, by applying the “ConvNet Trick” for approximating.


#Links for dataset:
image dataset : https://drive.google.com/open?id=1LXsqbbCxDVdvyMAyaWft1u5NcAc8Ftm6
text dataset : https://drive.google.com/open?id=1h0ZBym1VEU-dw0MSyxyVe8N3C9tz8MIn
