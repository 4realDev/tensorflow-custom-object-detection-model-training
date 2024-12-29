# (in scripts) python own-scripts\preprocessing\run_image_auto_labelling.py -imgp C:\Users\vbraz\Desktop\ML_Images_21-02-23 -mth 0.9
# IMPORTANT: running this script via console, make sure you are in scripts direcory!

# This library can be used to create image annotation XML files in the PASCAL VOC file format.
# fmt: off
import os
import sys

import config
import cv2
import numpy as np
# pip install pascal-voc-writer for XML Pascal Writer
from pascal_voc_writer import Writer
from PIL import Image

sys.path.insert(
    1, '..\..\scripts\own-scripts\sticky-notes-detection')

from miro_tfod_functions import (
    get_bounding_boxes_above_min_score_thresh, get_detections_from_img,
    load_latest_checkpoint_of_custom_object_detection_model)

files = config.files
paths = config.paths
min_score_thresh = config.min_score_thresh
bounding_box_and_label_line_thickness = config.bounding_box_and_label_line_thickness

import argparse

parser = argparse.ArgumentParser(
    description="Script to auto-generate labeled Pascal XML Files for images")
parser.add_argument("-imgp",
                    "--images_path",
                    help="Path to the folder where the images are stored.",
                    metavar='simages_path', 
                    type=str, 
                    default=None,
                    required=True)
parser.add_argument("-mth",
                    "--min_threshold",
                    metavar='min_threshold', 
                    help="Minimal Thredhold for the TensorFlow Object Detection inside the images.",
                    type=float, 
                    default=min_score_thresh)


args = parser.parse_args()

def auto_label_images(images_path: str, min_score_thresh: float):
    # get the path or directory
    for image_name in os.listdir(images_path):

        # check if the image ends with png or jpg or jpeg
        image_name_with_lowercase_ending = image_name.lower()
        if  image_name_with_lowercase_ending.endswith(".png") or \
            image_name_with_lowercase_ending.endswith(".jpg") or \
            image_name_with_lowercase_ending.endswith(".jpeg"):

            print("GENERATE LABELS FOR: ", image_name)
            image_path = os.path.join(images_path, image_name)
            image_xml_file_name = os.path.splitext(image_name)[0] + '.xml'

            img = cv2.imread(image_path)
            img_detections = get_detections_from_img(img)
            img_detection_data_list = get_bounding_boxes_above_min_score_thresh(
                detections=img_detections,
                imgHeight=img.shape[0],
                imgWidth=img.shape[1],
                min_score_thresh=min_score_thresh
            )

            # create pascal voc writer (image_path, width, height)
            writer = Writer(image_path, img.shape[1], img.shape[0])

            for img_detection_data in img_detection_data_list:
                # https://mlhive.com/2022/02/read-and-write-pascal-voc-xml-annotations-in-python
                # ::addObject(name, xmin, ymin, xmax, ymax)
                writer.addObject(
                    img_detection_data['classname'], 
                    img_detection_data['xmin'], 
                    img_detection_data['ymin'], 
                    img_detection_data['xmax'], 
                    img_detection_data['ymax']
                )

            # write to file
            try:
                writer.save(os.path.join(
                    images_path, image_xml_file_name))
                print(
                    f"SUCCESS - generated {image_xml_file_name} for image {image_name}")
                print(
                    "____________________________________________________________________________________________________\n")
            except Exception as e:
                print(
                    f"ERROR - Could not generate {image_xml_file_name} for image {image_name} - {e}")
                print(
                    "____________________________________________________________________________________________________\n")

    print("FINISH AUTO LABELING PROCESS")


def main():
    load_latest_checkpoint_of_custom_object_detection_model()
    auto_label_images(
        args.images_path,  
        args.min_threshold, 
    )


if __name__ == '__main__':
    main()
