from tensorflow.keras.layers import Input, Conv2D, Activation, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from scipy.misc import imread
import sol5_utils as sol5_utils
from scipy.ndimage.filters import convolve as convolve
from enum import Enum
import random
from skimage.color import rgb2gray
import model_parameters as parm

class Representation(Enum):
    RGB = 1
    GRAY = 2


def read_image(filename, representation):
    image = imread(filename)
    image_float = image.astype(np.float64)
    image_float /= parm.NORMALIZATION

    if representation == Representation.RGB:
        return rgb2gray(image_float)
    elif representation == Representation.GRAY:
        return image_float
    else:
        raise ValueError('Wrong representation argument')


def load_dataset(files, batch_size, corruption_func, crop_size):
    """
    :param files: list of clean images files.
    :param batch_size: the size of the batch for each iteration of Stochastic Gradient Descent.
    :param corruption_func: a function receiving a numpy’s array representation of an image as a single
    argument, returns a randomly corrupted version of the input image.
    :param crop_size: a tuple (height, width) specifying the crop size of the patches to extract.
    :return: a Python’s generator object which outputs random tuples with the form
    (source_batch, target_batch), where each output variable is an array of shape (batch_size, 1,
    height, width).
    """
    h, w = crop_size
    #memoization
    container = {}

    while True:
        source_batch = np.zeros((batch_size, parm.DIMENSION, h, w))
        target_batch = np.zeros((batch_size, parm.DIMENSION, h, w))

        for i in range(batch_size):
            file = np.random.choice(files)
            if file not in container:
                container[file] = read_image(file, Representation.RGB)
                image = container[file]
            else:
                image = container[file]
            corrupted_image = corruption_func(image)
            x = np.random.choice(image.shape[0] - h)
            y = np.random.choice(image.shape[1] - w)

            target_batch[i, 0, :, :] = image[x:x + h, y:y + w] - parm.SUBTRACT_VALUE
            source_batch[i, 0, :, :] = corrupted_image[x:x + h, y:y + w] - parm.SUBTRACT_VALUE
        yield (source_batch, target_batch)


def resblock(input_tensor, num_channels):
    first_layer = Conv2D(num_channels, (parm.CONV_HIGHT, parm.CONV_WIDTH), padding='same')(input_tensor)
    second_layer = Activation('relu')(first_layer)
    third_layer = Conv2D(num_channels, (parm.CONV_HIGHT, parm.CONV_WIDTH), padding='same')(second_layer)
    output_tensor = add([third_layer, input_tensor])
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    set_input = Input(shape=(parm.DIMENSION, height, width))
    layer = Conv2D(num_channels, (parm.CONV_HIGHT, parm.CONV_WIDTH), padding='same')(set_input)
    layer = Activation('relu')(layer)
    head_layer = layer

    for i in range(num_res_blocks):
        layer = resblock(layer, num_channels)
    addition_layers = add([layer, head_layer])
    unit_blocks = Conv2D(parm.DIMENSION, (parm.CONV_HIGHT, parm.CONV_WIDTH),
                         padding='same')(addition_layers)
    return Model(inputs=set_input, outputs=unit_blocks)


def train_model(model, images, corruption_func, batch_size,
                samples_per_epoch, num_epochs, num_valid_samples):
    """
    Divide the images into a training set and validation set, using an 80-20 split,
    generate from each set a dataset.
    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. assuming these paths are complete,
    and should append anything to them.
    :param corruption_func: a function receiving a numpy’s array representation of an image as a single
    argument, and returns a randomly corrupted version of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: the number of samples in each epoch.
    :param num_epochs: the number of epochs for which the optimization will run.
    :param num_valid_samples: the number of samples in the validation set to test on after every epoch.
    """
    edge = int(parm.PERCENT * len(images))
    split_one = images[:edge]
    split_two = images[edge:]
    crop_s = model.input_shape[2:4]
    training_set = load_dataset(split_one, batch_size, corruption_func, crop_s)
    validation_set = load_dataset(split_two, batch_size, corruption_func, crop_s)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=parm.BETA_2))
    model.fit_generator(training_set, steps_per_epoch=samples_per_epoch/batch_size, epochs=num_epochs,
                        validation_data=validation_set, validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    :param corrupted_image: a gray image and with values in the range if [0, 1]
     with type float64.
    :param base_model: the loaded trained model.
    :return: a neural network trained to restore small patches.
    """
    h = corrupted_image.shape[0]
    w = corrupted_image.shape[1]
    shape_input = Input(shape=(h, w, 1))
    shape_output = base_model(shape_input)
    new_model = Model(inputs=shape_input, outputs=shape_output)
    new_model.set_weights(base_model.get_weights())
    restored_image = new_model.predict(corrupted_image[np.newaxis, ..., np.newaxis] - parm.SUBTRACT_VALUE)[0]
    restored_image = np.clip(restored_image.reshape((1, h, w)) + parm.SUBTRACT_VALUE, 0, 1)
    return restored_image.astype(np.float64)[0]


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Creating a new image with gaussian noise.
    :param image: a gray image with values in the range of [0, 1] with type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian
    distribution.
    :param max_sigma:a non-negative scalar value larger than or equal to min_sigma,
    representing the maximal variance of the gaussian distribution.
    :return: image with gaussian noise.
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise_w_sigma = np.random.normal(0, sigma, image.shape)
    noise_image = image + noise_w_sigma

    return np.clip(noise_image, 0, 1).astype(np.float64)


def learn_denoising_model(num_res_blocks=parm.NUM_RES_BLOCKS, quick_mode=True):
    """
    Train a network which expect patches of size 16×16, and have 32 channels.
    :param num_res_blocks: the number of the residual blocks.
    :param quick_mode: use only 10 images in a batch, 30 samples per epoch, just 2 epochs and only 30
    samples for the validation set.
    :return: a trained denoised model.
    """
    num_channels = parm.NUM_CHANNELS * 2
    paths = sol5_utils.images_for_denoising()
    den_model = build_nn_model(parm.MODEL_HEIGHT_NOISE, parm.MODEL_WIDTH_NOISE, num_channels, num_res_blocks)
    corrption_func = lambda img: add_gaussian_noise(img, 0, parm.NOISE_FACTOR)
    if quick_mode:
        train_model(den_model, paths,
                    corrption_func, parm.BATCH_SIZE_SMALL,
                    parm.SAMPLES_PER_EPOCH_SMALL,
                    parm.SMALL_EPOCH, parm.VALID_SAMPLES_SMALL)
    else:
        train_model(den_model, paths,
                    corrption_func, parm.BATCH_SIZE,
                    parm.SAMPLES_PER_EPOCH_BIG, parm.BIG_EPOCH, parm.VALID_SAMPLES_BIG)
    return den_model


def add_motion_blur(image, kernel_size, angle):
    """
    Simulate motion blur on the given image using a square kernel with the size "kernel_size" where the line
    has the given angle in radians, measured relative to the positive horizontal axis.
    :param image:  a gray image with values in the range of [0, 1] with type float64.
    :param kernel_size:  an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians with the range [0, π).
    :return: image with motion blur.
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel).astype(np.float64)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    :param image: a gray image with values in the range of [0, 1] with type float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: image with random motion blur.
    """
    angle = np.random.uniform(0, np.pi)
    edge = len(list_of_kernel_sizes) - 1
    random_num = random.randint(0, edge)
    return add_motion_blur(image, list_of_kernel_sizes[random_num], angle).astype(np.float64)


def learn_deblurring_model(num_res_blocks=5, quick_mode=True):
    """
    Train a network which expect patches of size 16×16, and have 32 channels
    :param num_res_blocks: the number of the residual blocks.
    :param quick_mode: use only 10 images in a batch, 30 samples per epoch, just 2 epochs and only 30
    samples for the validation set.
    :return: a trained deblurring model
    """
    paths = sol5_utils.images_for_deblurring()
    deb_model = build_nn_model(parm.HEIGHT_BLUR_MODEL, parm.WIDTH_BLUR_MODEL, parm.NUM_CHANNELS, num_res_blocks)
    corrption_func = lambda image: random_motion_blur(image, [7])
    if quick_mode:
        train_model(deb_model, paths, corrption_func,
                    parm.BATCH_SIZE_SMALL, parm.SAMPLES_PER_EPOCH_SMALL,
                    parm.SMALL_EPOCH, parm.VALID_SAMPLES_SMALL)
    else:
        train_model(deb_model, paths, corrption_func,
                    parm.BATCH_SIZE, parm.SAMPLES_PER_EPOCH_BIG,
                    parm.BIG_EPOCH, parm.VALID_SAMPLES_BIG)
    return deb_model
