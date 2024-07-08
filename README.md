# AI Reliability in Industrial Safety: A Case Study with ABB IRB 1200 and Intel RealSense D415
![Project setup](imgs/project_image.png)

## Overview
This project explores the application of advanced AI techniques to enhance industrial safety with the ABB IRB 1200 robotic system and Intel RealSense D415 camera. Supervised by Mathias Verbeke and Matthias De Ryck at KU Leuven, Faculty of Engineering Technology, this work is a significant part of the academic curriculum for the Master of AI in Business and Industry, academic year 2023-2024.

## Project Description
The main objective of this study is to assess and improve the reliability of AI systems in industrial environments. By integrating computer vision and robotics, the project aims to enhance the accuracy and safety of industrial robots during operation.

## Methodology
The project follows a structured approach involving several key stages:
1. **Image Acquisition and Preprocessing**: Using the Intel RealSense D415, raw images are captured and processed to correct distortions and generate masks.
2. **3D Object Localization**: Objects are located in 3D space by transforming camera coordinates to robot base coordinates.
3. **Tool and Camera Calibration**: Continuous calibration updates to account for tool wear and camera shifts.
4. **Uncertainty Handling**: Techniques to manage and mitigate uncertainty in robot operations are explored, including runtime monitoring and predictive uncertainty estimation using neural networks.

## Key Components
- **U-Net for Image Segmentation**: Modified U-Net architecture to improve mask creation and object detection in noisy environments.
- **MLP Models for Predictive Analysis**: Two multilayer perceptrons (MLP) are developed to predict and adjust the robot's actions in real-time.
- **Evaluation Metrics**: Metrics such as accuracy, Dice score, and binary cross-entropy are used to measure the performance of the AI models.

## Results
Preliminary results indicate a high degree of accuracy in object localization and a promising ability to handle operational uncertainties in real-time.

## Future Work
Further research will focus on improving the depth perception of images, refining the machine learning models, and expanding the dataset for better generalization of the AI models in different industrial settings.

## Installation
Clone the repository to get started with the project:

## Usage
Details on how to set up and run the demonstrations are provided in the repository's documentation.

## Contributors
- Iñigo Aduna Alonso (Researcher)
- Mathias Verbeke (Supervisor)
- Matthias De Ryck (Co-Supervisor)
