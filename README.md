# Description of Files

### Baseline Model
This folder contains the files necessary to load the baseline model. The baseline model was trained with 1,386 images without preprocessing. The model was trained for 5,000 steps with a batch size of 32.

### Microsoft_Surfer_Baseline_Object_Detection 
This .ipynb file was created with Google Colab to use transfer learning and train our object detector. I am using Google Colab because of the free GPU service that will greatly improve the training time compared to the CPU on my laptop.

### ObjectDetection.py
This python script loads the object detection model and uses OpenCV to draw the bounding boxes around the detected objects. An .MP4 file is returned with the detections.

### [rename_and_flip_files.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/rename_and_flip_files.py)
This python script renames the training images and saves a reflected copy. The purpose of reflecting the image is so that the model can learn from an equal amount of "left" and "right" waves.

### split_images.py
This python script splits the image files into a train and test set along with their corresponding .XML files.

