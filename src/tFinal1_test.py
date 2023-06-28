"""
Mask R-CNN Coco
Run the model on test images
"""
import skimage
import tFinal1_model

model_list = tFinal1_model.load_models()

image = skimage.io.imread("test.jpg")
tFinal1_model.get_result(image, model_list)
#tFinal1_model.get_result(image, model_list, True)
