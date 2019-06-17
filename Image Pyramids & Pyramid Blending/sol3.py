import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scs
from scipy.misc import imread
import scipy.ndimage as ni
from skimage.color import rgb2gray
from scipy.signal import convolve2d as conv
import os

#------------------------------prv function-----------------------------------------#
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


def create_kernel(kernel_size):

    counter = kernel_size#counter for while loop
    kernel = np.array([1]).reshape(1, 1).astype('float64')#the first

    while (counter > 1):
        a = np.array([1, 1]).reshape(1, 2).astype('float64')
        kernel = (conv(kernel, a))#conveltion to get the binomial line
        counter = counter - 1
    gaus = kernel
    gaus = gaus / np.sum(gaus)  # normilaze the kernel
    return gaus
#------------------------------prv function-----------------------------------------#
##################################3.1################################################


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

def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    bulding laplacing pyramid by expanding and to subtract from the gaussian pyramid
    :param im: the original image
    :param max_levels: the max level we allowed for bulding the pyr
    :param filter_size: the size of our filter
    :return: filter vector of the gauss bin and an array of laplacian pyrmid
    '''
    ga_pyr,filter_vec= build_gaussian_pyramid(im,max_levels,filter_size)#getting the gaussian pyr
    pyr=[] #initalize our arrray
    for i in range(len(ga_pyr)-1):#creating the laplican pyrmid
        new_im= expand(ga_pyr[i],ga_pyr[i+1],filter_vec)
        my_lap=ga_pyr[i]-new_im# creating the laplican matrix
        pyr.append(my_lap)#adding to our array
    pyr.append(ga_pyr[-1]) #adding the lasone
    return pyr,filter_vec
#**********************************help function**************************************
def reduce(filter_vec,im):
    new_im=im
    new_im = ni.filters.convolve(new_im, filter_vec)  # bluring by x
    new_im = ni.filters.convolve(new_im, filter_vec.T)  # bluring by y
    new_im = new_im[::2, ::2]  # reducing
    return new_im

def expand(imB,imS,filter_vec):
    my_zeros = np.zeros(imB.shape)  # zero matrix in the shape of the g(i)
    my_zeros[::2, ::2] = imS  # the expand
    new_im = ni.filters.convolve(my_zeros, 2 * filter_vec)  # blure on x
    new_im = ni.filters.convolve(new_im, (2 * filter_vec).T)  # blure on y
    return new_im


#**********************************help function**************************************
##########################3.2######################################################


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''

    :param lpyr: our laplicina pyrmid
    :param filter_vec: our filter vecotre
    :param coeff: in the length of our pyrmid , muliply with our laplican and adding
    :return: image which created by adding the laplacian (after expand) to each other.

    '''
    my_len=len(lpyr)
    img=coeff[my_len-1]*lpyr[my_len-1]#my base image
    for i in range(my_len - 1):
        k=my_len-(i+1)
        img=expand(lpyr[k-1],img,filter_vec)#expanding
        img+=lpyr[k-1]*coeff[k-1]#summing
    return img


##############################3.3########################################

def render_pyramid(pyr, levels):
    '''

    :param pyr:the pyrmid we want to display
    :param levels: the amound of pictures to display
    :return:pictuer which holding the all pyramid
    '''
    hight= pyr[0].shape[0]#getting the hight
    length=0
    my_level=levels-1
    while my_level>=0:
        length+=pyr[my_level].shape[1]#getting the length
        my_level=my_level-1
    ren=np.zeros((hight,length))#our return
    col = 0
    for i in range(levels):
        my_stretch = (pyr[i] - np.min(pyr[i])) / (np.max(pyr[i]) - np.min(pyr[i]))#streching
        ren[:my_stretch.shape[0],col:col+my_stretch.shape[1]]= my_stretch#combine the pyramid
        col+=my_stretch.shape[1]
    return ren

def display_pyramid(pyr,levels):
    im=render_pyramid(pyr, levels)#getting the image we want to display
    plt.imshow(im, cmap=plt.gray())
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    '''

    :param im1:first image we want to blend
    :param im2:seconed image we want to blind
    :param mask:our mask
    :param max_levels: the max level we want to get
    :param filter_size_im: size of the filter for gaussian and laplican pyramid
    :param filter_size_mask: size of filter for the mask pyramid
    :return:blend image
    '''
    L1,v1=build_laplacian_pyramid(im1,max_levels,filter_size_im,)#laplcianan for first image
    L2,v2=build_laplacian_pyramid(im2,max_levels,filter_size_im,)#laplacian for seconed image
    mask=mask.astype('float64')#getting the mask
    Gm,vm=build_gaussian_pyramid(mask,max_levels,filter_size_mask)#getting the gaussian for mask
    pyr=[]
    for i in range(len(L1)):
        L3=Gm[i]*L1[i]+(1-Gm[i])*L2[i]#creating the blend pyramid
        pyr.append(L3)
    coeff=np.ones(len(L3))
    img=np.clip(laplacian_to_image(pyr,v1,coeff),0,1)#bulding the image
    return img


####################################################################################################
def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
    im1 = read_image(relpath('externals/im1.jpg'), 2)
    im2= read_image(relpath('externals/im2.jpg'), 2)
    mask=read_image(relpath('externals/mask.jpg'),1).astype(np.bool)
    blend_im=np.zeros(im1.shape)
    im1_R=im1[:,:,0]
    im1_G=im1[:,:,1]
    im1_B=im1[:,:,2]
    im2_R = im2[:, :, 0]
    im2_G = im2[:, :, 1]
    im2_B = im2[:, :, 2]
    blend_im[:,:,0],blend_im[:,:,1],blend_im[:,:,2]= pyramid_blending(im1_R,im2_R,mask,10,7,5),\
                                                     pyramid_blending(im1_G,im2_G,mask,10,7,5),\
                                                     pyramid_blending(im1_B,im2_B,mask,10,7,5)

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(blend_im)
    plt.show()
    return (im1, im2, mask, blend_im)


def blending_example2():
    im1 = read_image(relpath('externals/im11.jpg'), 2)
    im2= read_image(relpath('externals/im22.jpg'), 2)
    mask=read_image(relpath('externals/mask2.jpg'),1).astype(np.bool)
    blend_im=np.zeros(im1.shape)
    im1_R=im1[:,:,0]
    im1_G=im1[:,:,1]
    im1_B=im1[:,:,2]
    im2_R = im2[:, :, 0]
    im2_G = im2[:, :, 1]
    im2_B = im2[:, :, 2]
    blend_im[:,:,0],blend_im[:,:,1],blend_im[:,:,2]= pyramid_blending(im1_R,im2_R,mask,10,7,5),\
                                                     pyramid_blending(im1_G,im2_G,mask,10,7,5),\
                                                     pyramid_blending(im1_B,im2_B,mask,10,7,5)

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(blend_im)
    plt.show()
    return (im1, im2, mask, blend_im)







    
