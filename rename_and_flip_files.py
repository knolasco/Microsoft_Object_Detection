import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# set up our paths
CWD = os.getcwd()
IMAGE_PATH = os.path.join(CWD, 'allImages', 'all\\')
IMAGE_SAVE_PATH = os.path.join(CWD, 'allImages', 'modifiedImages\\')

if not os.path.isdir(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)

def rename_images(path, path_to_save):
    for ind, img in enumerate(os.listdir(path)):
        filename = 'img{}.JPG'.format(ind)
        img = mpimg.imread(path + img)
        img = Image.fromarray(img)
        img.save(path_to_save + filename)

def flip_images(path, path_to_save):
    for img in os.listdir(path):
        if '.JPG' in img:
            tmp_img = mpimg.imread(path + img)
            img_flipped = np.flip(tmp_img, 1)
            img_flipped = Image.fromarray(img_flipped)
            img_flipped.save(path_to_save + img.replace('.JPG','') + '_flip.JPG')

rename_images(IMAGE_PATH, IMAGE_SAVE_PATH)
flip_images(IMAGE_SAVE_PATH, IMAGE_SAVE_PATH)