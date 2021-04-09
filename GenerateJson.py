import os
import json
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from moviepy.editor import VideoFileClip, clips_array
import matplotlib.pyplot as plt

# set up paths
CWD = os.getcwd()
MODEL_PATH = os.path.join(CWD, 'model')
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'checkpoints')
CONFIG_PATH = os.path.join(MODEL_PATH, 'config\pipeline.config')
ANNOTATION_PATH = os.path.join(MODEL_PATH, 'annotations')
FRAME_PATH = os.path.join(CWD, 'frames_4231')
OUTPUT_PATH = os.path.join(CWD, 'output_json_4231')
OUTPUT_IMAGES_PATH = os.path.join(CWD, 'output_images_4231')

paths = [OUTPUT_PATH, OUTPUT_IMAGES_PATH]
for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)

SCORE_THRESHOLD = 0.6

# ---------------------------------------------------- helper functions --------------------------------------------------

@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def get_centroid_from_dims(box, size):
    """
    The coordinates of the box are in the 'box' variable.
    y_min = box[0], x_min = box[1], y_max = box[2], x_max = box[3]
    The coordinates that are inputted are normalized, so we multiply by the img width and height.
    Then, the centroid is in the center of the rectangle, which is the average of the x's and y's
    """
    width = size[0]
    height = size[1]
    left, right, top, bottom = box[1]*width, box[3]*width, box[2]*height, box[0]*height
    return (int((left + right)/2), int((top + bottom)/2))

class SurferDetection_withJSON():
    """
    this object will load the model, and run open cv to calculute the distance between a surfer and the wave zone.
    """
    def __init__(self, model_checkpoint):
        """
        We want to tell the class which model to use
        """
        self.model_checkpoint = model_checkpoint
        self.zone_location = None
        self.surfer_height = None
    
    def denormalize_dims(self, box):
        width = self.size[0]
        height = self.size[1]
        left, right, top, bottom = box[1]*width, box[3]*width, box[2]*height, box[0]*height
        return left, right, top, bottom


    def load_model(self):
        """
        We will load the model that is specified by the model path. The model path is the checkpoint that we want to use.
        """
        # load the trained model from checkpoint
        configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
        detection_model = model_builder.build(model_config = configs['model'], is_training = False)

        ckpt = tf.compat.v2.train.Checkpoint(model = detection_model)
        ckpt.restore(os.path.join(CHECKPOINT_PATH, self.model_checkpoint)).expect_partial()

        category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')
        self.detection_model = detection_model
        self.category_index = category_index

    def ready_image(self, image_name):
        """
        This will initialize the image we will use for the analysis. We will also prepare the output image
        """
        self.file_name = image_name
        self.file_path = OUTPUT_PATH
        self.img_out_path = OUTPUT_IMAGES_PATH
        
        image_path = os.path.join(FRAME_PATH, image_name)

        img = Image.open(image_path)
        width = int(img.size[0])
        height = int(img.size[1])
        size = (width, height)

        # output img
        output_img = image_name.replace('.jpg', '_centroid_detection.jpg')
        self.detection_img = os.path.join(self.img_out_path, output_img)

        # make the cap, size, and results attributes of the class
        self.img = img
        self.size = size

    def draw_lines_to_centroids(self, c1, c2):
        img = cv2.line(self.image_with_detections, c1, c2, (255,0,0), 5)
        return img
    
    def preprocess_image(self, image):
        """
        The baseline model was trained without preprocessing the images.
        Model 2 had preprocessed images that have PIL.ImageOps.grayscale and a blur applied.
        Model 3 had preprocessed images that have PIL.ImageOps.solarize and black and white applied
        """
        if self.model_checkpoint == 'ckpt-6': # baseline model, no preprocessing needed
            return image
        elif self.model_checkpoint == 'ckpt-9': # second model
            img = Image.fromarray(image)
            img = ImageOps.grayscale(img) # make grayscale
            img = img.filter(ImageFilter.BLUR) # apply blur
            img = img.convert('RGB') # need this to convert back to np
            img = np.asarray(img)
            return img
        else: # third model
            img = Image.fromarray(image)
            img = ImageOps.solarize(img) # inverts pixel values above the 128 threshold
            img = img.convert('LA') # make grayscale
            img = img.convert('RGB') # need this to convert back to np
            img = np.asarray(img)
            return img

    def add_to_json(self, dist = None, surfer_box = None, zone_box = None):
        """
        Save the distance between centroids and the bounding boxes
        """
        if surfer_box is not None:
            surfer_dict = {'left': surfer_box[0], 'right': surfer_box[1],'top': surfer_box[2],'bottom': surfer_box[3]}
        else:
            surfer_dict = {'left': None, 'right': None,'top': None,'bottom': None}
        if zone_box is not None:
            zone_dict = {'left': zone_box[0], 'right': zone_box[1],'top': zone_box[2],'bottom': zone_box[3]}
        else:
            zone_dict = {'left': None, 'right': None,'top': None,'bottom': None}

        json_dict = {'BoundingBox': {'surfer' : surfer_dict,'zone' : zone_dict},'Distance' : dist, 'ZoneLocation' : self.zone_location, 'SurferVerticalPosition': self.surfer_height}
        json_name = self.file_name.replace('.jpg', '.json')
        json_name = os.path.join(self.file_path, json_name)
        with open(json_name, 'w') as fp:
            json.dump(json_dict, fp, indent = 4)
    def label_location(self, c1, c2):
        surfer_x = c1[0]
        surfer_y = c1[1]
        box_x = c2[0]
        box_y = c2[1]

        if surfer_x - box_x < 0:
            self.zone_location = 'SurferLeft'
        else:
            self.zone_location = 'SurferRight'
        if surfer_y - box_y < 0:
            self.surfer_height = 'Top'
        else:
            self.surfer_height = 'Bottom'

    def calculate_distance(self, c1, c2, surfer_box, zone_box):
        """
        Calculate the euclidean distance between the two centroids.
        The first centroid is the surfer, the second is the zone.
        """
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        dist = np.linalg.norm(c1 - c2)
        self.label_location(c1, c2) # since there are two detections, find the relative location to each other
        self.add_to_json(dist, surfer_box, zone_box)
    
    def run_PIL(self):
        """
        This will run the opencv and display our results on the screen
        """
        no_detections = False
        two_detections = False
        image_np = np.array(self.img)
        image_np_processed = self.preprocess_image(image_np)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_processed, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor, self.detection_model)
                
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        self.image_with_detections = image_np_with_detections

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates= True,
            max_boxes_to_draw=2,
            min_score_thresh= SCORE_THRESHOLD,
            agnostic_mode=False)


        boxes = detections['detection_boxes'] # get the detection boxes and scores
        scores = detections['detection_scores']
        classes = detections['detection_classes'] + label_id_offset
        # keep only the boxes whose score is greater than 0.75
        boxes_filtered = [(box, box_class) for box, score, box_class in zip(boxes, scores, classes) if score > SCORE_THRESHOLD] 
            
        if len(boxes_filtered) > 1: # more than one detection
            two_detections = True
            if boxes_filtered[0][1] == 1: # the label for a surfer is 1
                surfer_box = boxes_filtered[0][0]
                zone_box = boxes_filtered[1][0]
            else:
                surfer_box = boxes_filtered[1][0]
                zone_box = boxes_filtered[0][0]

        elif len(boxes_filtered) == 1:
            self.surfer_height = None
            self.zone_location = None
            if boxes_filtered[0][1] == 1:
                surfer = True
                surfer_box = boxes_filtered[0][0]
                zone_box = None
            else:
                surfer = False
                zone_box = boxes_filtered[0][0]
                surfer_box = None
        else:
            self.surfer_height = None
            self.zone_location = None
            surfer_box = None
            zone_box = None
        
        if two_detections:
            surfer_centroid = get_centroid_from_dims(surfer_box, self.size)    
            zone_centroid = get_centroid_from_dims(zone_box, self.size)
            surfer_box = self.denormalize_dims(surfer_box)
            zone_box = self.denormalize_dims(zone_box)
            self.image_with_detections = self.draw_lines_to_centroids(surfer_centroid, zone_centroid)                       
            self.calculate_distance(surfer_centroid, zone_centroid, surfer_box, zone_box)
        else:
            if surfer_box is not None:
                surfer_box = self.denormalize_dims(surfer_box)
            if zone_box is not None:
                zone_box = self.denormalize_dims(zone_box)

            self.add_to_json(surfer_box = surfer_box, zone_box = zone_box)

        final_img = Image.fromarray(self.image_with_detections)
        final_img.save(self.detection_img)


def main():
    detector = SurferDetection_withJSON('ckpt-11')
    detector.load_model()
    for img in os.listdir(FRAME_PATH):
        detector.ready_image(img)
        detector.run_PIL()

if __name__ == '__main__':
    main()