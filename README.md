# AI Reliability in Industrial Safety: A Case Study with ABB IRB 1200 and Intel RealSense D415
![Project setup](github_imgs/Scheme.png)

## Overview
This project explores the application of advanced AI techniques to enhance industrial safety with the ABB IRB 1200 robotic system and Intel RealSense D415 camera. Supervised by Mathias Verbeke and Matthias De Ryck at KU Leuven, Faculty of Engineering Technology, this work is a significant part of the academic curriculum for the Master of AI in Business and Industry, academic year 2023-2024.

## Project Description
The main objective of this study is to assess and improve the reliability of AI systems in industrial environments. By integrating computer vision and robotics, the project aims to enhance the accuracy and safety of this industrial robot during operation. This project has been developed following the CRISP-DM (Cross Industry Standard Process for Data Mining) structure.

## Methodology
The project follows a structured approach involving several key stages:
1.- **Data Understanding**: Using the Intel RealSense D415, raw images are captured and processed to correct distortions and using an HSV filter the mask of the objects is generated. The data includes several components: images (visual representations captured by the camera), masks (binary maps that indicate the location and shape of the specific pieces the robot will have to pick) and ground truth labels (they provide the location of the masks in the robot frame). 

2.- **Data Preparation**: Speciffic features of the images and masks are identified, a binarization check is passed through the masks, dataset splitting, dataset augmentation with both spatial (Horizontal and vertical flips, shiftscalerotate) and image speciffic transforms (Random Brightness Contrast, Gaussian Noise, Coarse Dropout, Blurring) to enhance the model performance preventing overfitting and we add some other corruptions (Shot Noise, Impulse Noise, Defocus Blus, glass blur, motion blur, fog, brightness, contrast, elastic transform, speckle noise, gaussian noise, spatter, saturation increase). 

3.- **Modelling**: The first block corresponds to the U-net image segmentation model [U-Net Model](https://github.com/zhixuhao/unet.git) followed by a postprocessing block which take the segmentation of the image from the output of the U-Net model and enhance the segmentation. This is done by applying morphological operations, contour detection and the Watershed algorithm followed by an area treshold. After this a final filtering block is applied, which gets the features of each potential piece segmented by the first model, and compares the features to the features of the pieces in the healthy (non corrupted) dataset, this is done to prevent pieces to pass to the next block. After the previous blocks we get a list of the (x,y,z) position in the warped camera frame (WCF) of the pieces in the working range of the camera and robot, to send the coordinates to the robot frame we have to transform them into (x,y,z) position in the robot frame. For this, two different approaches are applied. The first approach consists on using the transformation matrixes acquires in the callibration process. In the second approach we train an mlp end to end that given the (x,y,z) position of the image in the warped camera frame, it outputs the (x,y,z) position in the robot frame. 

4.- **Evaluation**: The performance of each block is determined by corrupting the images with the different corruptions and changing the image sizes. 

5.- **Deployment**: After several iterations the end to end model has been tested in the robot ABB IRB 1200 using [Robot Demonstrator - Matthias de Ryck](https://github.com/MatthiasDR96/robot_demonstrator.git) as a baseline. 

## Results
The results are shown in the results folder.

## Future Work
Further research will focus on improving the depth perception of images, refining the machine learning models, and expanding the dataset for better generalization of the AI models in different industrial settings.

## Installation
For the instalation, use the following lines of code in your terminal. 
cd <Path to the folder you´d like to save this project>
conda create <new environment name>
conda activate <new environment name> python=3.8
pip install -r requirements.txt 
git clone <link to this repository>

## Notebooks


## Contributors
- Iñigo Aduna Alonso (Researcher)
- Mathias Verbeke (Supervisor)
- Matthias De Ryck (Co-Supervisor)
