import numpy as np
from scipy.misc import imread as imread
from scipy.signal import convolve2d as conv
from skimage.color import rgb2gray
import scipy.ndimage as ni


#function : return image at gray or RGB
def read_image(filename,representation): #filename-> the picture name,
    # representation 1 for gray 2 for RGB
    im =imread(filename)#catching the picture
    im_float=im.astype(np.float64)#changing to float 64 type
    im_float /=255 #normilaize
    if (representation==1): #turning gray and return
        im_g = rgb2gray(im_float)
        return im_g

    elif (representation==2): #return RGB
        return im_float

###########################################################################################33

#functinon: getting image and kernel size return blur image(going through low pass filter)
def blur_spatial(im,kernel_size):

    gaus=create_kernel_b(kernel_size)#creating kernel

    return conv(im,gaus,fillvalue=0,mode='same').astype("float64")#return the blur function,
    # edges padd with zeros, and in the same size

#functino : helper functino, geting size , return gauss kernel with this size*size
def create_kernel_b(kernel_size):

    counter = kernel_size#counter for while loop
    kernel = np.array([1]).reshape(1, 1)#the first

    while (counter > 1):
        a = np.array([1, 1]).reshape(1, 2)
        kernel = (conv(kernel, a))#conveltion to get the binomial line
        counter = counter - 1

    gaus = conv(kernel.T, kernel)#conv to get the kernel, size*size
    gaus = gaus / np.sum(gaus)#normilaze the kernel
    return gaus

######################################################################################3#########

def build_gaussian_pyramid(im, max_levels, filter_size):
    """creating gaussian pyrmid

    :param im: the image
    :param max_levels: the max level we allowed for bulding the pyr
    :param filter_size: the size of our filter
    :return:filter vector of the gauss bin and an array of gaussian pyrmid
    """
    pyr=[]#array of the pyr
    pyr.append(im)
    filter_vec= create_kernel(filter_size)#creating the filter
    ###reduce
    new_im=im
    imx_shape = new_im.shape[0]
    imy_shape =new_im.shape[1]
    my_max=max_levels
    while ((imx_shape>=32 and imy_shape>=32) and my_max>1):
        new_im= reduce(filter_vec,new_im)
        pyr.append(new_im)
        imx_shape = new_im.shape[0]
        imy_shape = new_im.shape[1]
        my_max-=1
    return pyr,filter_vec


def reduce(filter_vec, im):
    new_im = im
    new_im = ni.filters.convolve(new_im, filter_vec)  # bluring by x
    new_im = ni.filters.convolve(new_im, filter_vec.T)  # bluring by y
    new_im = new_im[::2, ::2]  # reducing
    return new_im

def create_kernel(kernel_size):
    counter = kernel_size  # counter for while loop
    kernel = np.array([1]).reshape(1, 1).astype('float64')  # the first

    while (counter > 1):
        a = np.array([1, 1]).reshape(1, 2).astype('float64')
        kernel = (conv(kernel, a))  # conveltion to get the binomial line
        counter = counter - 1
    gaus = kernel
    gaus = gaus / np.sum(gaus)  # normilaze the kernel
    return gaus
        
##########################################################################################################
