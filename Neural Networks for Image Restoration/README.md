**Overview**<br>
This exercise deals with neural networks and their application for image restoration. In this exercise 
I've developed a general workflow for training networks to restore corrupted images and then applied this
workflow on two different tasks: <br/>
1.Image denoising.<br/>
2.Image deblurring.<br/>

**Background**<br>
The method I've implemented consists of the following three steps:
1.Collecting “clean” images, applying simulated random corruptions and extracting small patches.
2.Training a neural network to map from corrupted patches to clean patches.
3.Given a corrupted image,I've used the trained network to restore the complete images by restoring each
patch separately, by applying the “ConvNet Trick” for approximating.

**Dataset Links:**<br>
Image dataset: https://drive.google.com/open?id=1LXsqbbCxDVdvyMAyaWft1u5NcAc8Ftm6 <br/>
Text dataset:  https://drive.google.com/open?id=1h0ZBym1VEU-dw0MSyxyVe8N3C9tz8MIn 

#### How to run?<br/>
I've wrapped the exercise with console application.<br/>
1.For image denoising, download the images from the image dataset to image_dataset\train folder inside the project.<br/>
2.For image deblurring, dowload the images from the text dataset to text_dataset\\train folder inside the project.<br/>
`pip install -r requirements.txt`<br/>
run `main.py`<br/>
<br/>
*Notes:*<br/> 
1.If you want to play with the models paremters you can go to model_parameters.py.<br/> 
2.If you want to train the models quicker, go to sol5.py and change the quick_mode argument in the functions learn_deblurring_model/learn_denoising_model to TRUE
