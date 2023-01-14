from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from mrcnn import utils
from mrcnn import visualize

sys.path.insert(0, r'C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\mrcnn')

from visualize import set_user_percentage


from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#import custom

# Root directory of the project
ROOT_DIR = r"C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


WEIGHTS_PATH = r"E:\weights-cyc\logs\object20221203T0146\mask_rcnn_object_0200.h5"   # change it



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 2  # Background + full body and laser

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9  


# Code for Customdataset class. Same code is present in custom.py file also
class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):

        self.add_class("object", 1, "full body")
        self.add_class("object", 2, "Dent")
        #self.add_class("object", 3, "Dent laser")


        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(r'C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\dataset\train\model1_update_json.json'))
        #print("annotations1 is :", annotations1)

        annotations = list(annotations1.values())  # don't need the dict keys
        #print("annotations is :", annotations)

        annotations = [a for a in annotations if a['regions']]
       
        
        
        # Add images
        for a in annotations:
            
           
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
           
            #print("objects:",objects)
            name_dict = {"full body": 1,"Dent": 2}
            num_ids = [name_dict[a] for a in objects]

            #print("numids",num_ids)
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object", 
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

        #print("objects is :", objects)

# Inspect the model in training or inference modes values: 'inference' or 'training'
TEST_MODE = "inference"
ROOT_DIR = r"C:\Users\Administrator\Desktop\MASK-RCNN\dataset"

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



# Load validation dataset
# Must call before using the dataset
CUSTOM_DIR = r"C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\dataset"
dataset = CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")
dataset.prepare()
#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
print("dataset.class_names_Total :", dataset.class_names)

config = CustomConfig()
#LOAD MODEL. Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load COCO weights Or, load the last model you trained
weights_path = WEIGHTS_PATH
# Load weights
#print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print("[INFO] loading Instence Segmentation Network from disk...")




#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


import glob
import time
from PIL import Image

import cropandstack
from cropandstack import predictions3
from numpy import asarray







def predictions(path_to_bmptojpg_image):
    for file in glob.glob(path_to_bmptojpg_image):
        start_time = time.time()
        mask_img = predictions3(file)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        #cv2.imshow('mask_img', mask_img)

        
        
        image1 = mpimg.imread(file)
        
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        main_img = image1
        #cv2.imshow('image1 before', image1)
        

        

        middle_image1 = image1[250:1149, 200:800]
        
        #cv2.imshow('middle_image1', middle_image1)
        
        
        #cropimg = image1
        ############ Second Model  function call

        #image1_resize = cv2.resize(image1, (512,512))
        #start_time = time.time()
        results1 = model.detect([middle_image1], verbose=0)
        # Display results
        ax = get_ax(1)
        #print(ax)
        r1 = results1[0]
        #print("r1 :",r1)
        #start_time = time.time()
        #print(type(image1))
        img2Pil, t_area = visualize.display_instances(middle_image1, r1['rois'], r1['masks'], r1['class_ids'],dataset.class_names, r1['scores'], ax=ax, title="Predictions1")



##        img = cv2.cvtColor(np.array(img2Pil), cv2.COLOR_BGR2RGB)
##        cv2.imshow('img', img)
        #cv2.waitKey(0)

        main_img[250:1149, 200:800] = img2Pil

        upper_img_roi = mask_img[0:300, 0:500]
        #upper_img_roi = cv2.cvtColor(upper_img_roi, cv2.COLOR_RGB2BGR)
        #print('type of upper_img_roi', type(upper_img_roi))
        #cv2.imshow('upper_img_roi', upper_img_roi)

        main_img[0:300, 250:750] = upper_img_roi

        #cv2.imwrite("uperimg.jpg", img)

        lower_img_roi = mask_img[300:400,0:500]
        #cv2.imshow('lower_img_roi', lower_img_roi)
        #lower_img_roi = cv2.cvtColor(lower_img_roi, cv2.COLOR_RGB2BGR)


        main_img[1150:1250,250:750]  = lower_img_roi
        #cv2.imshow('lower_img_roi', lower_img_roi)

        img2Pil = cv2.resize(main_img, (512,512))

        # if set_user_percentage <= t_area:
        #     print('total Dent Area of middle surface  %', t_area)
        # else:
        #     print('lower than user_percentage')
  

    

        end = time.time()
        print("Execution time of the program is %s seconds ---" % (end - start_time))
        cv2.imshow('main_img', img2Pil)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break   
        cv2.destroyAllWindows()




path_to_bmptojpg_image = r'C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\output\*.jpg'

print(predictions(path_to_bmptojpg_image))

















    

    


    

