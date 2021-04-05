import os
import pandas as pd
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from moviepy.editor import VideoFileClip, clips_array
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
mpl.use('tkagg')
import seaborn as sns

# set up paths
CWD = os.getcwd()
MODEL_PATH = os.path.join(CWD, 'model')
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'checkpoints')
CONFIG_PATH = os.path.join(MODEL_PATH, 'config\pipeline.config')
ANNOTATION_PATH = os.path.join(MODEL_PATH, 'annotations')
VIDEO_PATH = os.path.join(CWD, 'video')
OUTPUT_VIDEO_PATH = os.path.join(VIDEO_PATH, 'motion_detection_videos')

if not os.path.isdir(OUTPUT_VIDEO_PATH):
    os.mkdir(OUTPUT_VIDEO_PATH)

# ---------------------------- define all of the helper functions ------------------------
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

def add_label(val):
    if val < 0:
        return 'Right'
    else:
        return 'Left'

class SurferDetection():
    """
    this object will load the model, and run open cv to calculute the distance between a surfer and the wave zone.
    """
    def __init__(self, model_checkpoint):
        """
        We want to tell the class which model to use
        """
        self.model_checkpoint = model_checkpoint
        self.distance_df = pd.DataFrame(columns = ['frame','distance' , 'surfer_x','surfer_y','zone_x','zone_y'])
    
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

    def ready_video(self, video_name):
        """
        This will initialize the video we will use for the analysis. We will also prepare the output video
        """
        self.video_name = video_name
        self.file_name = video_name.replace('.MOV', '')
         # make a folder for this video to save everything associated with it in there
        self.file_path = os.path.join(OUTPUT_VIDEO_PATH, self.file_name)
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)
        
        video = os.path.join(VIDEO_PATH, video_name)

        cap = cv2.VideoCapture(video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = (width, height)

        # output video
        output_video_name = video_name.replace('.MOV', '_centroid_detection.mp4')
        self.detection_video = os.path.join(self.file_path, output_video_name)
        result = cv2.VideoWriter(self.detection_video,  
                                cv2.VideoWriter_fourcc(*'XVID'), 
                                20.0, size)

        # make the cap, size, and results attributes of the class
        self.cap = cap
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = size
        self.result = result

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

    def add_to_df(self, dist, c1, c2):
        """
        Save the distance between centroids and the x/y coordinates of each centroid into the dataframe
        """
        tmp_df = pd.DataFrame({'frame' : self.frame_counter, 
                               'distance' : dist,
                               'surfer_x' : c1[0],
                               'surfer_y' : c1[1],
                               'zone_x' : c2[0],
                               'zone_y' : c2[1]}, index = [self.frame_counter])
        self.distance_df = self.distance_df.append(tmp_df)

    def plot_distance(self):
        """
        We will show a plot of the distance over time after the video is finished.
        """
        # make all columns numeric
        for column in self.distance_df.columns:
            self.distance_df[column] = self.distance_df[column].astype(float)
        # the first plot is for distance overall
        g = sns.scatterplot(data = self.distance_df, 
                            x = 'frame', 
                            y = 'distance',
                            alpha = 0.5,
                            edgecolor = 'k',
                            linewidth = 0.5)

        g.set(xlim = (self.distance_df['frame'].min(), self.distance_df['frame'].max()))
        g.set(ylim = (0, self.distance_df['distance'].max()))
        g.set(ylabel = 'Distance from Surfer to Zone (pixels)')
        g.set(xlabel = 'Frame')
        plt.title('Distance Over Time')
        plt.tight_layout()
        dist_plot_path = os.path.join(self.file_path, 'distance_plot.png')
        plt.savefig(dist_plot_path)

    def plot_location(self):
        """
        I will also make a plot for how far a surfer is in front or behind the zone.
        This can be found by calculating the difference in the x-values of the surfer and zone.
        """
        plt.cla()
        # make all columns numeric
        for column in self.distance_df.columns:
            self.distance_df[column] = self.distance_df[column].astype(float)

        # the next plot is for surfer being in front of the wave or behind it
        self.distance_df['delta_x'] = self.distance_df['surfer_x'] - self.distance_df['zone_x']
        self.distance_df['location_to_zone'] = self.distance_df['delta_x'].apply(add_label)

        # I want a vertical line to be on the plot to seperate the rights and lefts
        plt.plot([0,0], [self.distance_df['frame'].min(), self.distance_df['frame'].max()], 'k-', markersize = 15)
        g = sns.scatterplot(data = self.distance_df, 
                            y = 'frame', 
                            x = 'delta_x', 
                            hue = 'location_to_zone', 
                            palette = 'mako', 
                            alpha = 0.5,
                            edgecolor = 'k',
                            linewidth = 0.5)

        g.set(ylim = (self.distance_df['frame'].min(), self.distance_df['frame'].max()))
        g.set(xlim = (self.distance_df['delta_x'].min(),self.distance_df['delta_x'].max())) # might change this
        g.set(xlabel = 'Location of Surfer to the Zone')
        g.set(ylabel = 'Frame')
        plt.title('Location of Surfer Relative to the Zone')
        plt.legend(loc = 'upper left')
        plt.tight_layout()
        location_plot_path = os.path.join(self.file_path, 'location_plot.png')
        plt.savefig(location_plot_path)


    def calculate_distance(self, c1, c2):
        """
        Calculate the euclidean distance between the two centroids.
        The first centroid is the surfer, the second is the zone.
        """
        c1 = np.asarray(c1)
        c2 = np.asarray(c2)
        dist = np.linalg.norm(c1 - c2)
        self.add_to_df(dist, c1, c2)
    
    def run_opencv(self):
        """
        This will run the opencv and display our results on the screen
        """
        self.frame_counter = 0 # to keep track of the frames
        while True: 
            ret, frame = self.cap.read()
            if ret:
                image_np = np.array(frame)
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
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=2,
                            min_score_thresh=.75,
                            agnostic_mode=False)


                boxes = detections['detection_boxes'] # get the detection boxes and scores
                scores = detections['detection_scores']
                classes = detections['detection_classes'] + label_id_offset
                # keep only the boxes whose score is greater than 0.75
                boxes_filtered = [(box, box_class) for box, score, box_class in zip(boxes, scores, classes) if score > 0.75] 

                # define difference of image
                delta = cv2.absdiff(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))

                if delta.sum() != 0: # there is a detection on the screen
                    if len(boxes_filtered) > 1: # more than one detection on the screen
                        if boxes_filtered[0][1] != boxes_filtered[1][1]: # make sure we are dealing with a wave and a surfer
                            if boxes_filtered[0][1] == 1: # the label for a surfer is 1
                                surfer_box = boxes_filtered[0][0]
                                zone_box = boxes_filtered[1][0]
                            else:
                                surfer_box = boxes_filtered[1][0]
                                zone_box = boxes_filtered[0][0]
                            
                            surfer_centroid = get_centroid_from_dims(surfer_box, self.size)
                            zone_centroid = get_centroid_from_dims(zone_box, self.size)
                            self.image_with_detections = self.draw_lines_to_centroids(surfer_centroid, zone_centroid)
                            self.calculate_distance(surfer_centroid, zone_centroid)
                        else:
                            self.add_to_df(0, (0,0), (0,0)) # nothing to save
                    else:
                        self.add_to_df(0, (0,0), (0,0)) # nothing to save
                else:
                    self.add_to_df(0, (0,0), (0,0)) # nothing to save

                cv2.imshow('Wave Motion Detection', cv2.resize(self.image_with_detections, (800,600)))


                self.result.write(cv2.resize(self.image_with_detections, self.size))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                self.frame_counter += 1
            else:
                break
        self.result.release()
        self.cap.release()
    
    def save_df(self):
        """
        The dataframe created from this analysis may be useful in the future, so we save it
        """
        df_name = self.video_name.replace('.MOV', '.csv')
        self.distance_df.to_csv(os.path.join(self.file_path, df_name))
    
    def make_animation(self, dist = True):
        if dist:
            data = self.distance_df
            def animate(i):
                tmp_data = data.iloc[:i]

                plt.cla()
                plt.xlim([self.distance_df['frame'].min(), self.distance_df['frame'].max()])
                plt.ylim([self.distance_df['distance'].min(), self.distance_df['distance'].max()])
                g = sns.scatterplot(data = tmp_data,
                                    x = 'frame',
                                    y = 'distance',
                                    alpha = 0.5,
                                    edgecolor = 'k',
                                    linewidth = 0.5)
                plt.title('Distance Over Time - Animated')
                plt.xlabel('Frame')
                plt.ylabel('Distance from Surfer to Zone (pixels)')

        else:
            data = self.distance_df
            # animation function.  This is called sequentially
            def animate(i):
                tmp_data = data.iloc[:i]

                plt.cla()
                plt.plot([0,0], [self.distance_df['frame'].min(), self.distance_df['frame'].max()], 'k-', markersize = 15)
                plt.xlim([self.distance_df['delta_x'].min(), self.distance_df['delta_x'].max()])
                plt.ylim([self.distance_df['frame'].min(), self.distance_df['frame'].max()])
                g = sns.scatterplot(data = tmp_data, 
                                    y = 'frame', 
                                    x = 'delta_x', 
                                    hue = 'location_to_zone', 
                                    palette = 'mako', 
                                    alpha = 0.5,
                                    edgecolor = 'k',
                                    linewidth = 0.5)

                plt.title('Location of Surfer Relative to the Zone - Animated')
                plt.xlabel('Location of Surfer to the Zone')
                plt.ylabel('Frame')
                plt.legend(loc = 'upper left')

        # make animation
        fig = plt.figure()
        ani = FuncAnimation(fig, animate, interval = 1, save_count = self.distance_df['frame'].max())
        plt.tight_layout()

        # save the animation
        writermp4 = animation.FFMpegWriter(fps=20)
        if dist:
            animation_name = self.video_name.replace('.MOV', '_DistAnimation.mp4')
            self.distance_animation = os.path.join(self.file_path, animation_name)
            ani.save(self.distance_animation , writer = writermp4)
        else:   
            animation_name = self.video_name.replace('.MOV', '_LocationAnimation.mp4')
            self.location_animation = os.path.join(self.file_path, animation_name)
            ani.save(self.location_animation,  writer = writermp4)
    
    def make_side_videos(self):
        """
        Show the generated videos side by side to visualize the distance in real time.
        """
        # open clips
        location_clip = VideoFileClip(self.location_animation)
        detection_clip = VideoFileClip(self.detection_video)

        # put them side by side
        final_clip = clips_array([[location_clip, detection_clip]])
        aggregated_video_name = os.path.join(self.file_path, 'aggregated_video.mp4')
        final_clip.write_videofile(aggregated_video_name)

def run_detections(checkpoint, video):
    """
    We run this function to initialize the class object and do everything else
    """
    detection = SurferDetection(checkpoint)
    detection.load_model()
    detection.ready_video(video)
    detection.run_opencv()
    detection.save_df()
    detection.plot_distance()
    detection.plot_location()
    detection.make_animation()
    detection.make_animation(dist = False)
    detection.make_side_videos()

# -------------------------------------- end of helper functions -----------------------------------------------

run_detections('ckpt-11', 'CH0I4295.MOV')   