import os
import xml.etree.ElementTree as ET
import shutil
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np 

# set up paths
CWD = os.getcwd()
ALL_IMG_PATH = os.path.join(CWD, 'allImages')
TEST_IMG_PATH = os.path.join(ALL_IMG_PATH, 'test')
TRAIN_IMG_PATH = os.path.join(ALL_IMG_PATH, 'train')
TEST_COPY = os.path.join(ALL_IMG_PATH, 'test_copy')
TRAIN_COPY = os.path.join(ALL_IMG_PATH, 'train_copy')

copy_paths = [TEST_COPY, TRAIN_COPY]
for path in copy_paths:
    if not os.path.isdir(path):
        os.mkdir(path)

# ----------------------- helper functions ---------------------------
DEGREES = 2.5

def rotate_coords(xmin, xmax, ymin, ymax, size, counter = False):
    """
    This function will rotate the coordinates around the center of the image.
    We use trig to find the new coordinates
    """
    center = (int(size[0]/2), int(size[1]/2))
    # define the points
    p1 = [xmin, ymax] # bottom left
    p2 = [xmin, ymin] # top left
    p3 = [xmax, ymax] # bottom right
    p4 = [xmax, ymin] # top right

    # points relative to the center
    p1 = [p1[0] - center[0], p1[1] - center[1]]
    p2 = [p2[0] - center[0], p2[1] - center[1]]
    p3 = [p3[0] - center[0], p3[1] - center[1]]
    p4 = [p4[0] - center[0], p4[1] - center[1]]

    points = [p1,p2,p3,p4]

    for point in points:
        if point[0] == 0: # so that we don't divide by zero
            point[0] = 1
        
        original_theta = np.arctan(point[1]/point[0]) # returns radians
        if counter:
            new_theta = original_theta + np.deg2rad(DEGREES) # convert to radians to use later
        else:
            new_theta = original_theta - np.deg2rad(DEGREES) # convert to radians to use later

        length = np.sqrt(point[0]**2 + point[1]**2)
        cos = np.cos(new_theta)
        sin = np.sin(new_theta)

        point = [int(length*cos) + center[0], int(length*sin) + center[1]] # convert to original coords after transforming
    
    # now we need to return the boundaries
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)

    return [xmin, xmax, ymin, ymax]


class Image_Augmentor():
    """
    This class will augment the images that we already have by either brightening, rotating clockwise or counterclockwise.
    """
    def __init__(self, img_path, save_path):
        """
        We make an instance for each img path.
        """
        self.img_path = img_path
        self.save_path = save_path
        self.images = [img for img in os.listdir(self.img_path) if '.JPG' in img]
        self.xml_files = [xml_files for xml_files in os.listdir(self.img_path) if '.xml' in xml_files]


    def move_xml(self, xml, modify = False, counter = False):
        """
        This function will rename and/or modify the xml
        """
        if not modify: # no need to modify xml when we brighten
            path_to_xml = os.path.join(self.img_path, xml)
            shutil.copy(path_to_xml, self.save_path) # copy the xml to the copy folder
            xml_in_copy = os.path.join(self.save_path, xml)
            xml_new = os.path.join(self.save_path, xml.replace('.xml','_brightened.xml'))
            os.rename(xml_in_copy, xml_new) # rename the file
        
        else:
            xml_path = os.path.join(self.img_path, xml)
            xml_tree = ET.parse(xml_path)
            root = xml_tree.getroot()
            # grab the original coordinates
            for box in root.iter('bndbox'):
                xmin = int(box.find('xmin').text) # this is how we get the location of the box
                xmax = int(box.find('xmax').text)
                ymin = int(box.find('ymin').text)
                ymax = int(box.find('ymax').text)
                # find the new coordinates
                if counter:
                    xml_location = os.path.join(self.save_path, xml.replace('.xml','_cc.xml'))
                    new_coords = rotate_coords(xmin, xmax, ymin, ymax, self.size, counter = True)
                else:
                    xml_location = os.path.join(self.save_path, xml.replace('.xml', '_clock.xml'))
                    new_coords = rotate_coords(xmin, xmax, ymin, ymax, self.size)
                
                # replace the coordinates
                box.find('xmin').text = str(new_coords[0])
                box.find('xmax').text = str(new_coords[1])
                box.find('ymin').text = str(new_coords[2])
                box.find('ymax').text = str(new_coords[3])
            
            xml_tree.write(xml_location)
            
                
    def brighten_and_save(self):
        """
        The first augment is a simple brighten.
        """
        for img, xml in zip(self.images, self.xml_files):
            img_name = img.replace('.JPG','_brightened.JPG')
            img_name = os.path.join(self.save_path, img_name) # we will save it into the copy folder
            tmp_img_path = os.path.join(self.img_path, img) # establish path to image
            img = Image.open(tmp_img_path)  # open the image
            # initialize an enhancer
            enhancer = ImageEnhance.Brightness(img)
            factor = 1.5 # to brighten the image
            img_out = enhancer.enhance(factor)
            # save the new image
            img_out.save(img_name)
            self.move_xml(xml)
    
    def rotate(self, img, counter = True):
        """
        We will rotate all of the original images counter-clockwise and clockwise by 2.5 degrees.
        We don't want the rotation to be too extreme, otherwise the images won't represent a realistic wave/surfer.
        """
        if counter:
            img_name = img.replace('.JPG','_cc.JPG') # cc is counter-clockwise
            tmp_img_path = os.path.join(self.img_path, img) # establist path to img
            img_name = os.path.join(self.save_path, img_name)
            img = Image.open(tmp_img_path) # open the image
            self.size = img.size
            img_rotated = img.rotate(DEGREES)
        else:
            img_name = img.replace('.JPG','_clock.JPG') # clockwise rotation
            tmp_img_path = os.path.join(self.img_path, img) # establist path to img
            img_name = os.path.join(self.save_path, img_name)
            img = Image.open(tmp_img_path) # open the image
            self.size = img.size
            img_rotated = img.rotate(-1*DEGREES)
        return img_rotated, img_name

    def rotate_and_save(self):
        """
        Next, we rotate counterclockwise and clockwise
        """
        for img, xml in zip(self.images, self.xml_files):
            for rotation in ['counter', 'clock']:
                if rotation == 'counter':
                    img_rotated, img_name = self.rotate(img)
                    img_rotated.save(img_name)
                    self.move_xml(xml, modify = True, counter = True)
                else:
                    img_rotated, img_name = self.rotate(img, counter = False)
                    img_rotated.save(img_name)
                    self.move_xml(xml, modify = True)

    def make_more_images(self):
        """
        This will call all functions to apply the transformations
        """
        self.brighten_and_save()
        self.rotate_and_save()


for img_path, save_path in zip([TRAIN_IMG_PATH, TEST_IMG_PATH], [TRAIN_COPY, TEST_COPY]):
    augmentor = Image_Augmentor(img_path, save_path)
    augmentor.make_more_images()