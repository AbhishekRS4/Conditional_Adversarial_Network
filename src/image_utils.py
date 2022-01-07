import os
import skimage
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.color import rgb2lab, lab2rgb

def resize_image(img_rgb, image_size=(320, 320)):
    img_resized = resize(img_rgb, image_size)
    return img_resized

def convert_lab2rgb(img_lab):
    img_rgb = lab2rgb(img_lab)
    return img_rgb

def convert_rgb2lab(img_rgb):
    img_lab = rgb2lab(img_rgb)
    return img_lab

def apply_image_ab_post_processing(img_ab):
    img_ab = img_ab * 110.
    return img_ab

def apply_image_l_pre_processing(img_l):
    img_l = (img_l / 50.) - 1
    return img_l

def concat_images_l_ab(img_l, img_ab):
    img_lab = np.concatenate((img_l, img_ab), axis=-1)
    return img_lab

def read_image_rgb(file_img):
    img_rgb = imread(file_img)
    return img_rgb

def save_image_rgb(file_img, img_arr):
    imsave(file_img, img_arr)
    return