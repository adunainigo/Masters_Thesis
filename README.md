# AI Reliability in Industrial Safety: A Case Study with ABB IRB 1200 and Intel RealSense D415
![Project setup](github_imgs/Scheme.png)

## Overview
This project investigates the application of advanced AI techniques to enhance industrial safety using the ABB IRB 1200 robotic system and the Intel RealSense D415 camera. Supervised by Mathias Verbeke and Matthias De Ryck at KU Leuven, Faculty of Engineering Technology, this work constitutes a significant part of the academic curriculum for the Master of AI in Business and Industry, academic year 2023-2024.

## Project Description
The primary objective of this study is to assess and improve the reliability of AI systems in industrial environments. By integrating computer vision and robotics, this project aims to enhance the accuracy and safety of the ABB IRB 1200 during its operations.

## Methodology
The project is developed following the CRISP-DM (Cross Industry Standard Process for Data Mining) methodology:

0.- **Business Understanding**: A general overview of the importance of the topic and approaches developed is available in the milestones folder.

| File Name | Content |
|-----------|---------|
| Summary Milestones.docx | Written overview of the project and results in each intermediate step. |
| Intermediate_presentation.pptx | General overview of the project. |

1.- **Data Understanding**: Raw images are captured using the Intel RealSense D415 and processed to correct distortions. An HSV filter is applied to generate object masks. The data consists of images, masks (binary maps indicating the location and shape of specific objects for the robot to pick), and ground truth labels (providing the location of the masks in the robot frame).

| File Name | Description |
|-----------|-------------|
| DataUnderstanding&Preparation.ipynb | Includes data visualization, feature extraction, binary mask verification, dataset splitting, and data augmentation. |
| ./data/images_rgb/* | Contains RGB images captured by the Intel RealSense D415 camera (Camera Frame). |
| ./data/imgs_depth/* | Contains depth images captured by the Intel RealSense D415 camera (Camera Frame). |
| ./data/imgs_warped/* | Contains RGB images as seen from above, in the Warped Camera Frame (WCF). |
| ./data/labels_gt_RF/* | Contains ground truth (x, y, z) positions of the pieces within the robot's working frame, as observed from the Robot Frame. |
| ./data/masks_warped/* | Contains masks of the pieces as seen from above, in the Warped Camera Frame (WCF). |
| ./data/calibration_images/* | Contains files with perspective transformation matrixes obtained during the callibration of the camera, feature stats obtained from the healthy dataset which is used in the filtering block, and the data scalers for the data normalization and scaling used in the Multilayer Perceptron . |

2.- **Data Preparation**: Specific features of the images and masks are identified. Masks undergo binarization checks, and the dataset is split and augmented with spatial and image-specific transformations (e.g., horizontal and vertical flips, brightness contrast adjustments, Gaussian blur).

| File Name | Content |
|-----------|---------|
| DataUnderstanding&Preparation.ipynb | Data Visualization, Feature obtaining, binary mask verification, dataset splitting, dataset augmentation. |

3.- **Modelling**: Two segmentation models are trained using U-Net [U-Net Model](https://github.com/zhixuhao/unet.git) and Mask R-CNN architectures. A Region of Interest (ROI) filtering block with HSV filtering and contour analysis is applied, followed by a final filtering block comparing features to those in a healthy (non-corrupted) dataset. The (x,y,z) positions of pieces in the working range are transformed to the robot frame using either transformation matrices from the calibration process or an end-to-end MLP model.

4.- **Evaluation**: Performance is assessed by corrupting images with various corruptions using [Imagenet-C](https://github.com/hendrycks/robustness.git) as a benchmark. Some functions have been modified since in this repository the filters are fixed to an image size of 64x64 px. The metrics used for evaluation are dice score and accuracy. 

| File Name | Content |
|-----------|---------|
| Modelling&Evaluation.ipynb | Model building blocks definition (U-Net, Postprocessing, Filtering, MLP, Transformation Matrix block), training, and evaluation of each building block.  |
| corruption_application/* | This folder demonstrates the effects of applying various severities of ImageNet-C corruptions on the image of a piece within the robot's workspace. |
| postprocessing_area_treshold/* | This folder shows a sweep performed for the Postprocessing&Filtering block with different pixel counts (Area threshold). |
| before_and_after_postprocessing_spyderplot/* | This folder illustrates how the evaluation metric (Dice score) varies depending on the type and severity of corruption applied, with or without the postprocessing and filtering block. |
| boxplot_histogram_model_mlp/* | This folder contains two types of graphs: histograms showing the percentage of pieces detected by the robot based on the allowable permissiveness (sigma), and a boxplot of the error in calculating the piece's position depending on permissiveness, corruption, and severity, using an MLP as the final block. |
| boxplot_histogram_model_transfmatrix/* | This folder shows the same as the previous row but using transformation and correction matrices as the final block. |


5.- **Deployment**: The end-to-end model is tested on the ABB IRB 1200 robot using the [Robot Demonstrator - Matthias de Ryck](https://github.com/MatthiasDR96/robot_demonstrator.git) as a baseline. 

| File Name | Content |
|-----------|---------|
| main_threaded.py | For the deployment of the model, substitute this file in the "Robot Demonstrator/scripts/main_threaded.py" - Matthias de Ryck GitHub repository.|
| Deployment_video | A video of the system in operation. [See Demonstration Video](results/Deployment_video.mp4)|


## Future Work
Future research will focus on improving depth perception in images, refining machine learning models, and expanding the dataset to enhance the generalization of AI models in diverse industrial settings.

## Installation
```bash
cd <Path to the folder you’d like to save this project>
conda create -n <new environment name> python=3.8
conda activate <new environment name>
pip install -r requirements.txt
git clone <link to this repository>
```

## Contributors
- Iñigo Aduna Alonso (Researcher)
- Mathias Verbeke (Supervisor)
- Matthias De Ryck (Supervisor)
