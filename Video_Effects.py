import os
from moviepy.editor import ImageSequenceClip, CompositeVideoClip
from moviepy.video.fx.all import crop
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

CWD = os.getcwd()
FRAMES = os.path.join(CWD, 'frames_for_video')
FRAME_CROPPED = os.path.join(CWD, 'cropped_frames')
JSON_FILES = os.path.join(CWD, 'output_json_for_video')

if not os.path.isdir(FRAME_CROPPED):
    os.mkdir(FRAME_CROPPED)



#------------------------- helper functions ----------------
def make_df_from_json(json_files):
    df = pd.DataFrame(columns = ['SurferLeft','SurferRight','SurferBottom','SurferTop'])
    for ind,file in enumerate(json_files):
        with open(file) as data_file:
            data = json.load(data_file)
            surfer_dict = {'SurferLeft' : data['BoundingBox']['surfer']['left'],
                        'SurferRight': data['BoundingBox']['surfer']['right'],
                        'SurferBottom' : data['BoundingBox']['surfer']['bottom'],
                        'SurferTop' : data['BoundingBox']['surfer']['top']}
            tmp_df = pd.DataFrame(surfer_dict, index = [ind])
            df = df.append(tmp_df)
    return df

def get_max_dims_from_df(df):
    # get the dims of each box
    df['width'] = df['SurferRight'] - df['SurferLeft']
    df['height'] = df['SurferTop'] - df['SurferBottom']
    # get the index with the max width and height
    max_width_ind = df['width'].idxmax(axis = 1)
    max_height_ind = df['height'].idxmax(axis = 1)
    # get the dims of the max area
    max_width, max_height = int(df['width'].iloc[max_width_ind]), int(df['height'].iloc[max_height_ind])
    # we will need the centroid, so I get that now
    df['centroid_x'] = (df['SurferRight'] + df['SurferLeft'])/2
    df['centroid_y'] = (df['SurferTop'] + df['SurferBottom'])/2
    return max_width, max_height

def crop_frames(frames, max_width, max_height, df):
    """
    for each frame, we will crop with the centroid of the surfer
    """
    frames = [frame for frame in os.listdir(frames)]
    for ind, frame in enumerate(frames):
        if df['centroid_x'].isna()[ind]: # no surfer detected, use the previous detection
            i = 1
            while df['centroid_x'].isna()[ind - i]:
                i += 1
            x, y = int(df['centroid_x'].iloc[ind - i]), int(df['centroid_y'].iloc[ind - i])
        else:
            x, y = int(df['centroid_x'].iloc[ind]), int(df['centroid_y'].iloc[ind])

        left = x - max_width/2
        right = x + max_width/2
        top = y + max_height/2
        bottom = y - max_height/2
        img = Image.open(os.path.join(FRAMES, frame))
        img_cropped = img.crop((left, bottom, right, top))
        img_cropped.save(os.path.join(FRAME_CROPPED, frame))


#-------------------------------------- end of helper functions -----------------------------------------

# gather the frames
frames = [os.path.join(FRAMES, frame) for frame in os.listdir(FRAMES)]
frames = sorted(frames)
# gather the json files
json_files = [os.path.join(JSON_FILES, json_file) for json_file in os.listdir(JSON_FILES)]

df = make_df_from_json(json_files)
max_width, max_height = get_max_dims_from_df(df)
crop_frames(FRAMES, max_width, max_height, df)
# gather cropped frames
cropped_frames = [os.path.join(FRAME_CROPPED, frame) for frame in os.listdir(FRAME_CROPPED)]
cropped_frames = sorted(cropped_frames)
# make clips
clip = ImageSequenceClip(frames, fps = 5)
cropped_clips = ImageSequenceClip(cropped_frames, fps = 5)
cropped_clips = cropped_clips.resize(2.5)
full_clip = CompositeVideoClip([clip,
                                cropped_clips.set_position((0,0))])
full_clip.write_videofile('auto_zoom.mp4')