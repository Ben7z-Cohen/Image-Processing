import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import scipy.ndimage as ni
from skimage.color import rgb2gray
from scipy.signal import convolve2d as conv
from enum import Enum
import os

NORMALIZATION = 255
MIN_Y_SHAPE_SIZE = 32
MIN_X_SHAPE_SIZE = 32


class Representation(Enum):
    RGB = 1
    GRAY = 2


def read_image(filename, representation):
    image = imread(filename)
    image_float = image.astype(np.float64)
    image_float /= NORMALIZATION

    if representation == Representation.RGB:
        return rgb2gray(image)

    elif representation == Representation.GRAY:
        return image_float
    else:
        raise ValueError('Wrong representation argument')


def create_kernel(kernel_size):
    counter = kernel_size
    kernel = np.array([1]).reshape(1, 1).astype('float64')

    while counter > 1:
        a = np.array([1, 1]).reshape(1, 2).astype('float64')
        kernel = (conv(kernel, a))
        counter = counter - 1
    return kernel / np.sum(kernel)


def build_gaussian_pyramid(image, max_levels, filter_size):
    pyr = [image]
    filter_vec = create_kernel(filter_size)
    imx_shape = image.shape[0]
    imy_shape = image.shape[1]
    max_value = max_levels

    while (imx_shape >= MIN_X_SHAPE_SIZE and imy_shape >= MIN_Y_SHAPE_SIZE) and max_value > 1:
        image = reduce(filter_vec, image)
        pyr.append(image)
        imx_shape = image.shape[0]
        imy_shape = image.shape[1]
        max_value -= 1
    return pyr, filter_vec


def build_laplacian_pyramid(image, max_levels, filter_size):
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(image, max_levels, filter_size)
    pyr = []

    for i in range(len(gaussian_pyramid) - 1):
        image = expand(gaussian_pyramid[i], gaussian_pyramid[i + 1], filter_vec)
        laplacian = gaussian_pyramid[i] - image
        pyr.append(laplacian)
    pyr.append(gaussian_pyramid[-1])
    return pyr, filter_vec


def reduce(filter_vec, im):
    image = im
    # blurring x and y axis
    image = ni.filters.convolve(image, filter_vec)
    image = ni.filters.convolve(image, filter_vec.T)
    # reducing
    return image[::2, ::2]


def expand(image_b, image_s, filter_vec):
    zeros_arr = np.zeros(image_b.shape)
    zeros_arr[::2, ::2] = image_s
    image = ni.filters.convolve(zeros_arr, 2 * filter_vec)
    return ni.filters.convolve(image, (2 * filter_vec).T)



def laplacian_to_image(lpyr, filter_vec, coeff):
    pyr_length = len(lpyr)
    base_image = coeff[pyr_length - 1] * lpyr[pyr_length - 1]

    for i in range(pyr_length - 1):
        k = pyr_length - (i + 1)
        base_image = expand(lpyr[k - 1], base_image, filter_vec)
        base_image += lpyr[k - 1] * coeff[k - 1]
    return base_image


def render_pyramid(pyr, levels):
    height = pyr[0].shape[0]
    length = 0
    my_level = levels - 1

    while my_level >= 0:
        length += pyr[my_level].shape[1]
        my_level = my_level - 1
    rendered_image = np.zeros((height, length))
    col = 0

    for i in range(levels):
        my_stretch = (pyr[i] - np.min(pyr[i])) / (np.max(pyr[i]) - np.min(pyr[i]))
        rendered_image[:my_stretch.shape[0], col:col + my_stretch.shape[1]] = my_stretch
        col += my_stretch.shape[1]
    return rendered_image


def display_pyramid(pyr, levels):
    image = render_pyramid(pyr, levels)
    plt.imshow(image, cmap=plt.gray())
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_image, filter_size_mask):
    laplacian1, v1 = build_laplacian_pyramid(im1, max_levels, filter_size_image, )
    laplacian2, v2 = build_laplacian_pyramid(im2, max_levels, filter_size_image, )
    mask = mask.astype('float64')
    gaussian_mask, vm = build_gaussian_pyramid(mask, max_levels, filter_size_mask)
    pyr = []

    for i in range(len(laplacian1)):
        laplacian_iter = gaussian_mask[i] * laplacian1[i] + (1 - gaussian_mask[i]) * laplacian2[i]
        pyr.append(laplacian_iter)

    coeff = np.ones(len(laplacian_iter))
    return np.clip(laplacian_to_image(pyr, v1, coeff), 0, 1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example(image1_path, image2_path, mask_path):
    im1 = read_image(relpath(image1_path), Representation.GRAY)
    im2 = read_image(relpath(image2_path), Representation.GRAY)
    mask = read_image(relpath(mask_path), Representation.RGB).astype(np.bool)
    blend_im = np.zeros(im1.shape)

    im1_red, im1_green, im1_blue = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
    im2_red, im2_green, im2_blue = im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]
    blend_im[:, :, 0], blend_im[:, :, 1], blend_im[:, :, 2] = \
        pyramid_blending(im1_red, im2_red, mask, 10, 7, 5), \
        pyramid_blending(im1_green, im2_green, mask, 10, 7, 5), \
        pyramid_blending(im1_blue, im2_blue, mask, 10, 7, 5)

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(blend_im)
    plt.show()


if __name__ == "__main__":
    example = int(input("Choose which example to run: 1 for Thor, 2 for Shark eating Tiger.\n"))
    if example == 1:
        blending_example('externals/im1.jpg', 'externals/im2.jpg', 'externals/mask.jpg')
    elif example == 2:
        blending_example('externals/im11.jpg', 'externals/im22.jpg', 'externals/mask2.jpg')
    else:
        print("The input you entered is incorrect, please run again.")
