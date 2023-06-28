"""
Mask R-CNN Coco
Detect 'beef_tartare', 'grilled_salmon', 'mussels', 'peking_duck', 'pizza', 'spaghetti_bolognese', and 'steak'
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class FoodConfig(Config):
    """
    Configuration for training on the food dataset.
    Derives from the base Config class and overrides some values.
    """
    NAME = "tFinal1_Group2"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 7
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.01


############################################################
#  Dataset
############################################################

class FoodDataset(utils.Dataset):

    def load_food(self, dataset_dir, subset):
        """
        Load a subset of the Food dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        food_list = ['beef_tartare', 'grilled_salmon', 'mussels', 'peking_duck', 'pizza', 'spaghetti_bolognese', 'steak']
        food_num = 1
        for food_item in food_list:
            self.add_class("food", food_num, food_item)
            food_num+=1

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "data_final1_group2.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons = []
            objects = []
            for r in a['regions']:
                polygons.append(r['shape_attributes'])
                objects.append(r['region_attributes'])
            class_ids = []
            for n in objects:
                class_ids.append(n['Food'])
            print(class_ids)

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image("food", image_id=a['filename'], path=image_path, width=width, height=height,
                           polygons=polygons, class_ids=class_ids)
                
    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "food":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """
        Return the path of the image.
        """
        info = self.image_info[image_id]
        if info["source"] == "food":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FoodDataset()
    dataset_train.load_food(os.path.abspath("../../tFinal"), "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FoodDataset()
    dataset_val.load_food(os.path.abspath("../../tFinal"), "val")
    dataset_val.prepare()

    config = FoodConfig()
    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model_inference = modellib.MaskRCNN(mode="inference", config=FoodConfig(),
                                        model_dir=DEFAULT_LOGS_DIR)

    mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model,
                                                                            model_inference, dataset_val,
                                                                            calculate_map_at_every_X_epoch=1, verbose=1)

    print("Training network heads")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads', custom_callbacks=[mean_average_precision_callback])

############################################################
#  Training
############################################################

if __name__ == '__main__':
    print('Train')
    config = FoodConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    COCO_WEIGHTS_PATH = '../../mask_rcnn_coco.h5'
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    train(model)

