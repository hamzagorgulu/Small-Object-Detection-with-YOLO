# Small-Object-Detection-with-YOLO

This repository contains a research project aimed at designing a real-time small object detector based on the YOLO algorithm for use on edge and mobile devices. The project proposes modifying the YOLO architectures to improve its performance and accuracy in small object detection by removing the final layer of the backbone and adding an upsampled layer to the neck of the architecture. The modified YOLO architecture will be trained and evaluated using the VisDrone dataset and the inference time will be measured on an edge device. The repository includes scripts for applying random cropping to the training data, partitioning during inference, checking the labels in a dataset and testing the trained weights.


<details>
<summary>Abstract</summary>

The research on object detection keep their importance since it has many real world applications. When it comes to the detection of small objects, the current methods often face limitations such as lack of focus on small objects, scarcity of image samples with such objects, very small relative size in high resolution images, being prone to occlu- sion, limited context information, and vanishing spatial information in deep layers. This project aims to design a real-time small object detector based on the YOLO algorithm for use on the edge and mobile devices. To that end, we propose modifying the YOLO architectures to improve its performance and accuracy in small object de- tection. This involves removing the final layer of the backbone, which almost always captures the large objects, and adding an upsampled layer to the neck of the architecture to enhance small object detection accuracy. Additionally, we slice the input into smaller parts before feeding the net- work and will apply random cropping to annota- tions in order to increase the generalization of our model. The modified YOLO architecture will be trained and evaluated using the VisDrone dataset, which consists of a diverse collection of images and video clips captured by drone-mounted cam- eras. In addition, we will measure the inference time of our model on an edge device using a deployment framework such as TVM or TensorRT.
  
</details>


## Setup
Install official yolov5 repo to your local environment and create an environment with yolov5 requirements. Then apply cropping with crop_bulk.py to your dataset. Here are the definitions for the scripts:

- *crop_bulk.py*: This script applies random crop over the training data in order to increase the diversity of the dataset. When a random crop is applied to an image, the labels in this crop are also updated and saved in a new txt file. Basically, the model will pretend it as a new image during training. If you set the number of random crops as 10, you will have a 10x greater dataset.

- *detect_partition.py*: This file is the modified detect.py file. Additionally, it implements partitioning during the inference in order to increase the spatial resolution of the small objects in the test samples. 

- *control_yolo_dataset.py*: check if the labels in a dataset are correct. You will define the path of the dataset in the python file. You will need to put the images and the txt file for labels in the same folder. You can do this manually. There are probably different ways to make this but this works. Put them in the same folder and give this path. After running the script, you will the images with their labels as pop up, use “R” for a random sample, use “D” for forward and “A” for backward, and “Q” for quit. 

- *test.py*: You should give the trained weights to this script and it will give you the results of metrics. 


Here you can see the difference between default YOLO model and our proposed model. It is clearly able to detect very small object while the default YOLO model cannot. 
<img width="983" alt="image" src="https://user-images.githubusercontent.com/67511748/213193328-8380b3ec-235d-4af5-9fb9-6f64286a36cc.png">





The details of the architectural changes will be published soon. 

