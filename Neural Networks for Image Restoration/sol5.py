from keras.layers import Input, Convolution2D, Activation,merge
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from scipy.misc import imread as imread
import sol5_utils as sol5_utils
from scipy.ndimage.filters import convolve as convolve
import random
from skimage.color import rgb2gray



SUBSTRACT_VALUE = 0.5
NGRAY = 1
NCOLOER = 2
MAXPIX = 255
NUMCHANNELS = 32
NUMRESBLOCKS = 5
PERCENT = 0.8
BETA_2 = 0.9
MODELHEIGHT= 24
MODELWIDTH = 24
HEIGHTBLUR = 16
WIDTHBLUR = 16
BATCHSIZESMALL = 10
BACTHSIZEBIG = 100
SAMPLESPEREPOCHSMALL = 30
SAMPLEOEREPOCHBIG = 100000
EPOCHSSMALL = 2
EPOCHSBIG = 10
VALIDSAMPLESSMALLE = 30
VALIDSAMPLEBIG = 10000 
DIM = 1
CONVHIGHT = 3
CONVWIDTH = 3
NOISEFACTOER = 0.2

def read_image(filename, representation): 
    #1 for gray 2 for RGB
    im = imread(filename)  
    im_float = im.astype(np.float64) 
    im_float /= MAXPIX 
    if (representation == NGRAY): 
        im_g = rgb2gray(im_float)
        return im_g
    elif (representation == NCOLOER):  
        return im_float



def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    create data set
    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single
    argument, and returns a randomly corrupted version of the input image
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract
    :return:a Python’s generator object which outputs random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size, 1,
    height, width)
    """
    h,w = crop_size
    container = {}
    while True:
        source_batch = np.zeros((batch_size, DIM, h, w))
        target_batch = np.zeros((batch_size, DIM, h, w))
        for i in range(batch_size):
	        #chosing randomly filename
            myfile = np.random.choice(filenames)
            if (myfile not in container):
                container[myfile] = read_image(myfile, DIM)
                my_im = container[myfile]
            else:
                my_im = container[myfile]
	        #creating corrupt image
            corpt_im = corruption_func(my_im)
            # getting range for random patch
            x = np.random.choice(my_im.shape[0] - h)
            y = np.random.choice(my_im.shape[1] - w)
            #create random patch
            target_batch[i, 0, :, :] = my_im[x:x + h, y:y + w] - SUBSTRACT_VALUE
            source_batch[i, 0, :, :] = corpt_im[x:x + h, y:y + w] - SUBSTRACT_VALUE
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    """
    creating residual block
    :param input_tensor:  symbolic input tensor
    :param num_channels: number of channels for each of its  layers
    :return: symbolic output tensor of the layer configuration
    """

    #Convoltion - First layar
    first_l = Convolution2D(num_channels, CONVHIGHT, CONVWIDTH, border_mode='same')(input_tensor)

    #Applying Activation function RELU on our first layar

    second_l = Activation('relu')(first_l) 

    # Convolution - RELU Apllied Layar
    third_l = Convolution2D(num_channels, CONVHIGHT, CONVWIDTH, border_mode='same')(second_l)

    # output : third_l + input
    output_tensor = merge([third_l, input_tensor], mode ='sum')

    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):

    """
    creates the network model
    :param height: Keras model height
    :param width: Keras model width
    :param num_channels: Keras model number of channels
    :param num_res_blocks: number of res blocks
    :return: an untrained Keras model with input dimension the shape
    of , and all convolutional layers with number of
    output channels equal to num_channels
    """
    #Input for keras
    set_input = Input(shape = (DIM, height, width))
    layer = Convolution2D(num_channels, CONVHIGHT, CONVWIDTH, border_mode='same')(set_input)
    layer = Activation('relu')(layer)
    head_layer = layer
    for i in range(num_res_blocks):
        layer = resblock(layer, num_channels)
    addition_layers = merge([layer, head_layer], mode='sum')
    unit_blocks = Convolution2D(DIM, CONVHIGHT, CONVWIDTH, border_mode='same')(addition_layers)
    model = Model(input=set_input, output=unit_blocks)
    return model


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """
    divide the images into a training set and validation set, using an 80-20 split,generate from each set a dataset
    :param model: a general neural network model for image restoration.
    :param images: – a list of file paths pointing to image files. You should assume these paths are complete,
    and should append anything to them.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single
    argument, and returns a randomly corrupted version of the input image
    :param batch_size:  the size of the batch of examples for each iteration of SGD
    :param samples_per_epoch: The number of samples in each epoch
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch
    """
    #edge for split
    edge = int(PERCENT * len(images))
    split_one = images[:edge]
    split_two = images[edge:]
    crop_s = model.input_shape[2:4]
    training_set = load_dataset(split_one, batch_size, corruption_func, crop_s)
    validation_set = load_dataset(split_two, batch_size, corruption_func, crop_s)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=BETA_2))
    model.fit_generator(training_set, samples_per_epoch = samples_per_epoch, nb_epoch = num_epochs,
                        validation_data=validation_set, nb_val_samples = num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    restore full images
    :param corrupted_image: a grayscale image of shape  and with values in the [0, 1] range of type float64
    :param base_model:
    :return:  a neural network trained to restore small patches
    """


    h = corrupted_image.shape[0]
    w = corrupted_image.shape[1]
    s_input = Input(shape=(DIM,h,w))
    s_output = base_model(s_input)
    new_model = Model(input=s_input,output=s_output)
    new_model.set_weights(base_model.get_weights())
    restored_image = new_model.predict(corrupted_image[np.newaxis,np.newaxis,...] - SUBSTRACT_VALUE)[0]
    restored_image = np.clip(restored_image.reshape((1,h,w)) + SUBSTRACT_VALUE, 0, 1)
    return restored_image.astype(np.float64)[0]



def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    creating new image with gaussian noise
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian
    distribution.
    :param max_sigma:a non-negative scalar value larger than or equal to min_sigma, representing the maximal
    variance of the gaussian distribution.
    :return: image with gaussian noise
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise_w_sigma = np.random.normal(0, sigma, image.shape)
    noise_image = image + noise_w_sigma
    #image with gaussian noise
    corpt = np.clip(noise_image, 0, 1).astype(np.float64)
    return corpt 


def learn_denoising_model(num_res_blocks=NUMRESBLOCKS, quick_mode=False):
    """
    train a network which expect patches of size 24×24, using 48 channels
    :param num_res_blocks:  number of res blocks
    :param quick_mode: indication for quick mode or not
    :return: a trained denoising model
    """
    num_channels = NUMCHANNELS * 2
    paths = sol5_utils.images_for_denoising()
    #getting denoise model
    den_model = build_nn_model(MODELHEIGHT, MODELWIDTH,num_channels, num_res_blocks)
    corrpt_func = lambda img: add_gaussian_noise(img, 0, NOISEFACTOER)
    if quick_mode:
        train_model(den_model, paths, corrpt_func,  BATCHSIZESMALL, SAMPLESPEREPOCHSMALL,
         EPOCHSSMALL, VALIDSAMPLEBIG)
    else:
        train_model(den_model, paths, corrpt_func, BACTHSIZEBIG,
         SAMPLEOEREPOCHBIG, EPOCHSBIG, VALIDSAMPLEBIG)
    return den_model


def add_motion_blur(image, kernel_size, angle):
    """
    simulate motion blur on the given image using a square kernel of size kernel_size where the line
    has the given angle in radians, measured relative to the positive horizontal axis.
    :param image:  a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return: image with motion blur
    """
    my_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    corrpt = convolve(image, my_kernel).astype(np.float64)
    return corrpt


def random_motion_blur(image, list_of_kernel_sizes):
    """
    samples an angle at uniform from the range [0, π), and chooses a kernel size at uniform from the list
    list_of_kernel_sizes, followed by applying the previous function with the given image and the randomly
    sampled parameters.
    :param image:  a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: adds random motion blur to image.
    """
    #getting random angle
    angle = np.random.uniform(0, np.pi)
    edge = len(list_of_kernel_sizes) - 1
    # random index
    i = random.randint(0, edge)
    corrpt = add_motion_blur(image, list_of_kernel_sizes[i], angle).astype(np.float64)
    return corrpt


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    train a network which expect patches of size 16×16, and have 32 channels
    :param num_res_blocks: number of res blocks
    :param quick_mode: use only 10 images in a batch, 30 samples per epoch, just 2 epochs and only 30
    samples for the validation set.
    :return: a trained deblurring model
    """
    num_channels = NUMCHANNELS
    paths = sol5_utils.images_for_deblurring()
    #getting deblure model
    deb_model = build_nn_model(HEIGHTBLUR, WIDTHBLUR, num_channels,num_res_blocks)
    corrpt_func = lambda image: random_motion_blur(image, [7])
    if quick_mode:
        #training the model quick mode
        train_model(deb_model, paths, corrpt_func, BATCHSIZESMALL, SAMPLESPEREPOCHSMALL,
         EPOCHSSMALL, VALIDSAMPLEBIG)
    else:
        train_model(deb_model, paths, corrpt_func, BACTHSIZEBIG, SAMPLEOEREPOCHBIG, EPOCHSBIG, VALIDSAMPLEBIG)
    return deb_model




