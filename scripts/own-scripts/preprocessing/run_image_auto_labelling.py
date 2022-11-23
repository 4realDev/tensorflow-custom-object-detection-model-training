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
    1, '..\scripts\own-scripts\sticky-notes-detection')

from miro_tfod_functions import (
    get_bounding_boxes_above_min_score_thresh, get_detections_from_img,
    get_image_with_overlayed_labeled_bounding_boxes,
    load_latest_checkpoint_of_custom_object_detection_model,
    scan_for_object_in_video)

files = config.files
paths = config.paths
min_score_thresh = config.min_score_thresh
bounding_box_and_label_line_thickness = config.bounding_box_and_label_line_thickness


def auto_label_images(images_path: str, min_score_thresh: float):
    # get the path or directory
    for image_name in os.listdir(images_path):

        # check if the image ends with png or jpg or jpeg
        if (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
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
        paths['AUGMENTED_IMAGE_PATH'],  
        min_score_thresh, 
    )


if __name__ == '__main__':
    main()
