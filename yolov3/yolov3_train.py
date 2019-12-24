import cv2
import sys
import os
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import fileinput
import time
import numpy as np
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
from matplotlib import image as mpltimage
from matplotlib import pyplot as plt

from imageai.Detection.Custom import DetectionModelTrainer

def main():
    if (len(sys.argv) < 3):
        print("Usage: " + sys.argv[0] + " <dataset_dir> <pre_trained_model>")
        exit(-1)

    dataset_dir = sys.argv[1]
    pre_trained_model = sys.argv[2]
    
    if (len(sys.argv) > 2):
        image_name = sys.argv[2]
    else:
        print("Enter image or directory name: ")
        image_name = input()

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=dataset_dir)
    trainer.setTrainConfig(
        object_names_array=["Ripe Strawberry", "Green Strawberry", "Bad Strawberry"], 
        batch_size=4, 
        num_experiments=40, 
        train_from_pretrained_model=pre_trained_model)
    trainer.trainModel()

main()