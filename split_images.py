import os
import random

# set up our paths
CWD = os.getcwd()
MODIFIED_IMAGES_PATH = os.path.join(CWD, 'allImages' ,'modifiedImages')
TRAIN_PATH = os.path.join(CWD, 'allImages', 'train')
TEST_PATH = os.path.join(CWD, 'allImages', 'test')

# get a list of all of the jpgs
all_images = [file_path for file_path in os.listdir(MODIFIED_IMAGES_PATH) if '.xml' not in file_path]
# shuffle the images
random.shuffle(all_images)

# we'll use 85% of the images as training and the remaining as testing
train_size = int(0.85*len(all_images))
train_jpgs = all_images[:train_size]
test_jpgs = all_images[train_size:]

def move_images(img_paths, save_path):
    """
    This function will move all files from the list to the appropriate folder.
    """
    for img_path in img_paths:
        os.rename(os.path.join(MODIFIED_IMAGES_PATH, img_path), os.path.join(save_path, img_path))

def grab_and_move_xml(img_paths, save_path):
    """
    This function will grab the appropriate xml files and move it to the appropriate folder.
    """
    for img_path in img_paths:
        img_path = img_path.replace('.JPG', '.xml')
        os.rename(os.path.join(MODIFIED_IMAGES_PATH, img_path), os.path.join(save_path, img_path))

move_images(train_jpgs, TRAIN_PATH)
move_images(test_jpgs, TEST_PATH)
grab_and_move_xml(train_jpgs, TRAIN_PATH)
grab_and_move_xml(test_jpgs, TEST_PATH)