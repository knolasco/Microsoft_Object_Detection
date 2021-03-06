![MicrosoftTeams-image](https://user-images.githubusercontent.com/66560796/114232143-47fa5480-9930-11eb-9241-3fad9dd6b7b8.png)

# Introduction

This project was a collaboration with a team of machine learning engineers at Microsoft as part of a private Microsoft Hackathon. The purpose of this project was to build a prototype software that Olympic Surf coaches could use to analyse their athelete's techniques in real time. My contribution to this project was a model that would detect the "critical section" of the wave and the surfer on the wave, and an auto-zoom feature that keeps the camera focused on the surfer. Above is a screenshot of the prototype software that was completed at the end of the Hackathon.

# Description of Files

### [Video_Effects.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/Video_Effects.py)
This python script takes the json output from GenerateJson.py and automatically zooms into the surfer.
## [Example of Auto-Zoom](https://user-images.githubusercontent.com/66560796/114231182-f3a2a500-992e-11eb-992f-f5476d4f9799.mp4)

### [Microsoft_Surfer_Detection_3.ipynb](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/Microsoft_Surfer_Detection_3.ipynb)
This Google Colab notebook was used to train the final model for object detection. The pretrained model was trained with 1,386 images. The final checkpoint is saved in the [checkpoints](https://github.com/knolasco/Microsoft_Object_Detection/tree/main/Model3/checkpoints) folder of [Model3](https://github.com/knolasco/Microsoft_Object_Detection/tree/main/Model3)

### [ObjectDetection3.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/ObjectDetection.py)
This python script loads the object detection model and uses OpenCV to draw the bounding boxes around the detected objects. An .MP4 file is returned with the detections.

### [SurferDistance.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/SurferDistance.py)
This python script loads the object detection model and uses OpenCV to draw the bounding boxes around the detected objects. The output of this script is an aggregated video of the original clip with detections and the location of the surfer relative to the wave zone. The script also produces a .csv file containing the distance from the zone to the surfer over time. Finally, the script outputs two figures: an overall distance over time, and position from surfer to zone over time. [Here](https://user-images.githubusercontent.com/66560796/113612978-65b07c80-9605-11eb-9b9e-1df7b76228ea.mp4) is an example of the aggregated video.

### [rename_and_flip_files.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/rename_and_flip_files.py)
This python script renames the training images and saves a reflected copy. The purpose of reflecting the image is so that the model can learn from an equal amount of "left" and "right" waves.

### [split_images.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/split_images.py)
This python script splits the image files into a train and test set along with their corresponding .XML files.


### [DataAug.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/DataAug.py)
This python script augments the images used for training by brightening and slightly rotating. This is useful in the future in case we need to train the model on more data.

### [GenerateJson.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/GenerateJson.py)
This python script takes frames from a video as input and returns a json file with information about the bounding boxes for the surfer and wave. The jsons that are generated from this script are used to assemble a video that combines the result from another model.
