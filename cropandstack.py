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
from mrcnn2 import utils
from mrcnn2 import visualize
from mrcnn2.visualize import display_images
from mrcnn2.visualize import display_instances
import mrcnn2.model as modellib
from mrcnn2.model import log
from mrcnn2.config import Config
from mrcnn2 import model as modellib, utils
import tensorflow.compat.v1 as tf

from numpy import asarray
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#import custom

# Root directory of the project
ROOT_DIR = r"C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\cropmain"

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


WEIGHTS_PATH = r"E:\weights-cyc-ring\logs\object20221211T1709\mask_rcnn_object_0200.h5"   # change it



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 6  # Background + full body and laser

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 120

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9   


# Code for Customdataset class. Same code is present in custom.py file also
class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):

        self.add_class("object", 1, "vp-ring")
        self.add_class("object", 2, "vp-ring dent")
        self.add_class("object", 3, "shroud")
        self.add_class("object", 4, "shroud dent")
        self.add_class("object", 5, "footring")
        self.add_class("object", 6, "footring dent")
        #self.add_class("object", 3, "Dent laser")


        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations1 = json.load(open(r'C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\cropmain\dataset\train\12-11-22_model2_json.json'))
        #print("annotations1 is :", annotations1)

        annotations = list(annotations1.values())  # don't need the dict keys
        #print("annotations is :", annotations)

        annotations = [a for a in annotations if a['regions']]
       
        
        
        # Add images
        for a in annotations:
            
           
            polygons = [r['shape_attributes'] for r in a['regions']] 
            objects = [s['region_attributes']['names'] for s in a['regions']]
           
            #print("objects:",objects)
            name_dict = {"vp-ring": 1, "vp-ring dent": 2, "shroud":3, "shroud dent":4, "footring":5, "footring dent":6}
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
ROOT_DIR = r"C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\cropmain\dataset"

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



# Load validation dataset
# Must call before using the dataset
CUSTOM_DIR = r"C:\Users\MSI 1\OneDrive\Desktop\instence_segmentation\cropmain\dataset"
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

from PIL import Image
import io
from io import StringIO
import glob
import time


camera0 = True
camera1 = True
camera2 = True
camera3 = True

from matplotlib import pyplot as plt   
from numpy import asarray

##def find_coardinates_list(path):
##    f = open(path, "r")
##    data = json.loads(f.read())
##    #print(data)
##    a_key = "rect"
##
##    list_v = [a_dict[a_key] for a_dict in data]
##    
##    
##    cord=[int(i) for i in list_v[0].split(",")]
##    x1,y1,x2,y2=cord[0],cord[1],cord[2],cord[3]
##
##    return x1,y1,x2,y2


###############

def predictions3(file):
    
    
##    i = 0
##    if camera0 == True :
    img = cv2.imread(file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imshow('img_new',img )
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
##    x1,y1,x2,y2=find_coardinates_list('UserRects.txt')

    
    upper_body = img[0:300, 250:750]
    #cv2.imshow('upper_body',upper_body ) #blue
    img2 = img.copy()

    img2[0:300, 250:750] = upper_body 
##        print("upper_body", upper_body.shape)
    lower_body = img[1150:1270,250:750]
    #cv2.imshow('lower_body',lower_body ) #blue
    

    img2[1150:1270,250:750]=lower_body

    full_img = np.concatenate((upper_body, lower_body), axis= 0)
    full_img_c = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
##    full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
    #cv2.imshow('full_img_c_crop',full_img_c )
    #cv2.imwrite('testing_crop_img.jpg', full_img_c)

##    plt.imshow(full_img)
##    plt.show()

    start_time = time.time()
    results1 = model.detect([full_img_c], verbose=0)

    # Display results
    ax = get_ax(1)
    #print(ax)
    r1 = results1[0]
    #print('r1 is ', r1)

    ftc = (r1["class_ids"]==5)
    #print('object_count is ', object_count)
 #   for i in object_count:
    # print('i is ', i)
    ft_t_a = sum((sum(r1["masks"][:, :, ftc]*1)))
    ftd = (r1["class_ids"]==6)

    ft_d_a = sum((sum(r1["masks"][:, :, ftd]*1)))

    find_dant_area = round(((sum(ft_d_a)*100)/sum(ft_t_a)),2)

    print('find_dant_area is ', find_dant_area)




    # area = np.reshape(r1['masks'], (-1, r1['masks'].shape[-1])).astype(np.float32).sum()

    # print('area of mask ', area)

    #print(type(image1))
    img1 = visualize.display_instances(full_img_c, r1['rois'], r1['masks'], r1['class_ids'],dataset.class_names, r1['scores'], ax=ax, title="Predictions1")
    ##print("{} detections: {}".format(det_count, np.array(dataset.class_names)[det_class_ids]))
    

    

    imgNumpy = asarray(img1)

    #cv2.imshow("imgNumpy",imgNumpy)
    #cv2.waitKey(0)

    return imgNumpy





    
    
    
##fn=r'C:\Users\MSI 1\OneDrive\Desktop\cropandpaste\1\102412801_resized.jpg'
##
##predictions3(fn)











    

    


    

