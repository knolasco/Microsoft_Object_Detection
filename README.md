# Description of Files

### [Baseline Model](https://github.com/knolasco/Microsoft_Object_Detection/tree/main/basline_model/model)
This folder contains the files necessary to load the baseline model. The baseline model was trained with 1,386 images without preprocessing. The model was trained for 5,000 steps with a batch size of 32.

#### [Example of Results from Baseline Model](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/basline_model/CH0I4193_detection_Trim.gif)

### [Model 2](https://github.com/knolasco/Microsoft_Object_Detection/tree/main/Model2)
This folder contains the files necessary to load the second model. The second model was trained with 1,386 images with processing. I processed using blur and grayscale in the PIL module. The model was trained for 8,000 steps with a batch size of 64. I manually stopped training at 8,000 because the loss was starting to plateau.

#### [Example of Results from Model 2](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/Model2/CH0I4193_detection_Trim.gif)

### [Microsoft_Surfer_Baseline_Object_Detection](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/Microsoft_Surfer_Baseline_Object_Detection.ipynb) 
This .ipynb file was created with Google Colab to use transfer learning and train our object detector. I am using Google Colab because of the free GPU service that will greatly improve the training time compared to the CPU on my laptop.

### [Microsoft Surfer Detection 2](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/Microsoft_Surfer_Detection_2.ipynb)
This .ipynb file was created with Google Colab to use transfer learning adn train our object detector. This time, the images were preprocessed with gray scale and blur. The training was manually stopped because the performance was plateauing.

### [ObjectDetection.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/ObjectDetection.py)
This python script loads the object detection model and uses OpenCV to draw the bounding boxes around the detected objects. An .MP4 file is returned with the detections.

### [rename_and_flip_files.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/rename_and_flip_files.py)
This python script renames the training images and saves a reflected copy. The purpose of reflecting the image is so that the model can learn from an equal amount of "left" and "right" waves.

### [split_images.py](https://github.com/knolasco/Microsoft_Object_Detection/blob/main/split_images.py)
This python script splits the image files into a train and test set along with their corresponding .XML files.

