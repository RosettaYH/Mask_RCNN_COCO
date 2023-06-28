"""
Mask R-CNN - Inspect Trained Model
"""
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from termcolor import colored

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn import visualize
import mrcnn.model as modellib

from src import tFinal1_group0, tFinal1_group1, tFinal1_group2, tFinal1_group3, tFinal1_group4, tFinal1_group5, tFinal1_group6, tFinal1_group7


def load_models():

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    group0_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group0_0048.h5")  #37.36%
    group1_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group1_0047.h5")  #34.05%
    group2_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group2_0049.h5")  #48.93%
    group3_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group3_0050.h5")  #46.79%
    group4_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group4_0046.h5")  #56.47%
    group5_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group5_0050.h5")  #39.45%
    group6_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group6_0049.h5")  #43.39%
    group7_weights_path = os.path.join(MODEL_DIR, "mask_rcnn_tfinal1_group7_0047.h5")  #51.96%

    weights_path_list = [group0_weights_path, group1_weights_path, group2_weights_path, group3_weights_path, group4_weights_path, group5_weights_path, group6_weights_path, group7_weights_path]

    config_group0 = tFinal1_group0.FoodConfig()
    config_group1 = tFinal1_group1.FoodConfig()
    config_group2 = tFinal1_group2.FoodConfig()
    config_group3 = tFinal1_group3.FoodConfig()
    config_group4 = tFinal1_group4.FoodConfig()
    config_group5 = tFinal1_group5.FoodConfig()
    config_group6 = tFinal1_group6.FoodConfig()
    config_group7 = tFinal1_group7.FoodConfig()
    
    config_list = [config_group0, config_group1, config_group2, config_group3, config_group4, config_group5, config_group6, config_group7]
    
    DEVICE = "/cpu:0"
    TEST_MODE = "inference"
    
    with tf.device(DEVICE):
        model_group0 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[0])
        model_group1 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[1])
        model_group2 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[2])
        model_group3 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[3])
        model_group4 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[4])
        model_group5 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[5])
        model_group6 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[6])
        model_group7 = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_list[7])
    model_list = [model_group0, model_group1, model_group2, model_group3, model_group4, model_group5, model_group6, model_group7]
    
    for i in range(len(model_list)):
        print("Loading weights ", weights_path_list[i])
        model_list[i].load_weights(weights_path_list[i], by_name=True)
        
    return model_list

def get_result(image, model_list, show_image=False):
    food_0 = ['BG', 'group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']
    food_1 = ['BG', 'apple_pie', 'beignets', 'cannoli', 'cheesecake', 'chocolate_cake', 'creme_brulee', 'red_velvet_cake', 'tiramisu']
    food_2 = ['BG', 'beef_tartare', 'grilled_salmon', 'mussels', 'peking_duck', 'pizza', 'spaghetti_bolognese', 'steak']
    food_3 = ['BG', 'bibimbap', 'clam_chowder', 'hot_and_sour_soup', 'lobster_bisque', 'miso_soup', 'pho', 'pancakes']
    food_4 = ['BG', 'caesar_salad', 'caprese_salad', 'edamame', 'french_fries', 'fried_calamari', 'fried_rice', 'omelette']
    food_5 = ['BG', 'chicken_quesadilla', 'crab_cakes', 'deviled_eggs', 'donuts', 'escargots', 'macarons', 'scallops']
    food_6 = ['BG', 'breakfast_burrito', 'club_sandwich', 'croque_madame', 'grilled_cheese_sandwich',  'lasagna', 'lobster_roll_sandwich', 'waffles']
    food_7 = ['BG', 'dumplings', 'gyoza', 'ice_cream',  'oysters', 'samosa', 'spring_rolls', 'takoyaki']

    food_list = [food_0, food_1, food_2, food_3, food_4, food_5, food_6, food_7]
    
    results = model_list[0].detect([image])

    r = results[0]
    if (show_image):
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], food_list[0], r['scores'], ax=get_ax(1), title="Initial Predictions")
    
    c = []
    for i in range(len(r["rois"])):
        c.append([r['class_ids'][i], r['rois'][i]])
      
    index = 0
    for data in c:
        index+=1
        groupNum = data[0]
        coor = data[1]

        img1 = image[coor[0]:coor[2],coor[1]:coor[3]]
        results = model_list[groupNum].detect([img1])
        class_names = food_list[groupNum]

        r = results[0]
        if (show_image):
            visualize.display_instances(img1, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=get_ax(1), title="Predictions")

        final_result = {}
        for i in range(len(r["class_ids"])):
            
            class_id = r['class_ids'][i]
            score = r['scores'][i]*100
            label = class_names[class_id]
            if label not in final_result.keys():
                final_result[label] = score
            else:
                if (score > final_result[label]):
                    final_result[label] = score
            caption = "{}: {:.2f}%".format(label, score)
        
        print("\nArea", index, colored(final_result, 'blue'))

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
