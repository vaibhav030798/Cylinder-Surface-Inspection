# Cylinder-Surface-Inspection
Cylinder surface inspection with the help of instance segmentation MASK- RCNN

# Cylinder Surface Inspection System Documentation

**Author:** Vaibhav Kurrey

## Introduction

This documentation serves as a comprehensive guide to the development of the Cylinder Inspection System. The system's primary purpose is to identify defects in cylinders, such as surface dent marks, read date codes using Optical Character Recognition (OCR) technology, and verify the presence of safety caps on cylinder tops. Additionally, it includes an efficient alert mechanism to promptly notify operators of any detected defects.

## Tools and Technology

- IDE and code editors used for this project: Visual Studio 2019 and Visual Studio Code.
- Languages used for development: C# (.NET Framework 4.8) and Python (3.6).
- Front end of the project is developed and designed within Visual Studio 2022 using WinForms .NET framework.
- Event handling, API consuming, and camera control are done with C#.
- Database management of this project is done with the help of PostgreSQL, which has been created locally on the system.

## Camera Control

The cameras used in this project are 6 HikVision area scan cameras, which can be controlled with C# in the following way:

1. Install MVS SDK application into the system.
2. Sample code can be found in the location "C:\Program Files (x86)\MVS\Development".
3. Refer to the basic demo sample to get a basic understanding of how things are working.
4. Add the MvCameraControl.Net DLL into your project and use the required code to start and control the camera.
5. For this project, two classes have been created to control the cameras and keep the code separate, which are `cameraInstances.cs` and `HRcam.cs`.

## Consuming Python API with C#

A Python API has been created using FastAPI. When the project starts, it runs the Python API using command-line arguments and sends requests to test if the API is running successfully or not. If the API is not running on the first request, a thread tries to start it again. This process repeats until the API is running successfully. After successfully running the API, another request is sent to load the deep learning model into the Python API.

## System/Hardware Specifications

- CPU: Intel i9 12th generation
- GPU: Nvidia 3060-Ti
- RAM: 32GB
- Storage: 2TB (1TB SSD, 1TB HDD)

## Instance Segmentation on Custom Datasets in Windows 11

Instance segmentation is a cutting-edge computer vision technique that not only detects objects in an image but also precisely outlines and classifies each individual object. This level of granularity allows machines to understand images in a highly detailed manner, making it invaluable in various applications, including autonomous vehicles, medical imaging, and robotics.

### Backbone Architecture (ResNet101)

The core architecture used as the backbone for this project is ResNet101, a deep neural network known for its exceptional performance in image-related tasks.

## GitHub Repository

The project's source code is available on GitHub at [https://github.com/ahmedfgad/Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2).

## Research Paper

For in-depth understanding, the research paper associated with this project can be found at [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870).

## CUDA Requirements

- CUDA 11.x: This project relies on CUDA for GPU acceleration. You must install CUDA version 11.x.
  - CUDA Download Link: [CUDA 11.x Download](https://developer.nvidia.com/cuda-11.0-download-archive)

## cuDNN

cuDNN is a crucial GPU-accelerated library for deep neural networks. Make sure to install cuDNN.
- cuDNN Download Link: [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-archive)

## Visual Studio 2019

Visual Studio 2019 is the integrated development environment used for this project. Ensure you have it installed.
- Visual Studio 2019 Download Link: [Visual Studio 2019 Download](https://visualstudio.microsoft.com/downloads/)

## Setup Details

To set up the project environment and get it up and running, follow these step-by-step instructions:

**Step 1: Create a Conda Virtual Environment with Python 3.8**

First, create a clean Conda virtual environment with Python 3.8. This environment will isolate your project and its dependencies from your system's global Python installation.

**Step 2: Install the Dependencies**

Once your virtual environment is activated, install the necessary project dependencies. This ensures that your environment has all the required libraries and packages to run the project smoothly.

**Step 3: Clone the Mask_RCNN Repository**

Clone the Mask_RCNN repository from the GitHub repository you provided. This will give you access to the project's source code and assets.

**Step 4: Install pycocotools**

Install the pycocotools package, which is essential for working with COCO datasets and annotations. It's a critical part of the project's functionality.

**Step 5: Download the Pre-trained Weights**

Download the pre-trained weights required for the project. These pre-trained weights are trained on a large dataset and serve as a starting point for your model.

**Step 6: Test It**

Finally, after completing all the previous steps, test the project setup to ensure that everything is working as expected. This typically involves running some sample code or scripts to verify that the system, dependencies, and pre-trained weights are correctly integrated.

## Anaconda Environment Setup

To set up your project environment using Anaconda, follow these steps:

- Download Anaconda from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).
- Create environments using Anaconda:

conda create -n "yourenvname" python=3.8.0

conda activate yourenvname

conda install numpy scipy Pillow cython matplotlib

git clone https://github.com/philferriere/cocoapi.git

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI


- Download the Pre-trained Weights from [https://github.com/matterport/Mask_RCNN/releases](https://github.com/matterport/Mask_RCNN/releases).

## Creating a Dataset Using VGG Annotator (VIA)

To create a dataset for the project, follow these steps:

1. Download VIA (VGG Image Annotator) from [https://www.robots.ox.ac.uk/~vgg/software/via/](https://www.robots.ox.ac.uk/~vgg/software/via/).

2. Create a folder named "Dataset."

3. In the "Dataset" folder, create two subfolders: "train" and "val." Put training images in the "train" folder and validation images in the "val" folder.

## Step-by-Step Detections

- **Anchor Sorting and Filtering:** Visualizes every step of the first stage Region Proposal Network and displays positive and negative anchors along with anchor box refinement.

- **Bounding Box Refinement:** This is an example of final detection boxes (dotted lines) and the refinement applied to them (solid lines) in the second stage.

- **Mask Generation:** Examples of generated masks. These then get scaled and placed on the image in the right location.

- **Layer Activations:** Often it's useful to inspect the activations at different layers to look for signs of trouble (all zeros or random noise).

- **Weight Histograms:** Another useful debugging tool is to inspect the weight histograms. These are included in the `inspect_weights.ipynb` notebook.

- **Logging to TensorBoard:** TensorBoard is another great debugging and visualization tool. The model is configured to log losses and save weights at the end of every epoch.

## Python Scripts for Training and Testing

- `model.py` file: A script for loading RESNET101 architecture and model layer creation.
- `custom.py` file: A script for training custom datasets.
- `visualize.py` file: A script for visualizing the final result.
- `Cylinder_surface_Inspection.py` file: A script for image Segmentation and Mask.

## Building a Custom OCR Using YOLO (You Only Look Once) in Windows 11

### What is OCR?

OCR stands for Optical Character Recognition. It is used to read text from images such as a scanned document or a picture. This technology is used to convert virtually any kind of images containing written text (typed, handwritten, or printed) into machine-readable text data. Here, we are going to build an OCR that only reads the information you want it to read from a given image.

OCR has two major building blocks:

1. Text Detection
2. Text Recognition

Our first task is to detect the required text from images. Detecting the required text is a tough task but thanks to deep learning, we’ll be able to selectively read text from an image.

In deep learning, we are using YOLOv4 mainly because:

- It is the fastest when it comes to speed.
- It has good enough accuracy for our application.
- YOLOv4 has Feature Pyramid Network (FPN) to detect small objects better.

### YOLOv4

YOLO is a state-of-the-art, real-time object detection network. YOLOv4 is the fastest version of YOLO and uses Darknet-53 as its feature extractor. It has overall 137 convolutional layers, hence the name ‘Darknet-53.conv137’. It has successive 3 × 3 and 1 × 1 convolutional layers and has some shortcut connections.

For the purpose of classification, independent logistic classifiers are used with the binary cross-entropy loss function.

### Training YOLO using the Darknet framework

We will use the Darknet neural network framework for training and testing. The framework uses multi-scale training, lots of data augmentation, and batch normalization. It is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

You can find the source on GitHub at [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

Here is how to install the Darknet framework (If you are going to use GPU, then update GPU=1 and CUDNN=1 in the makefile):

Tools
CUDA Requirements: Cuda 11.x & Cudnn
