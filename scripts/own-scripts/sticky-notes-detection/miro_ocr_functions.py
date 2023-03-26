# TEST WITH PADDLEOCR
# Docs: https://github.com/PaddlePaddle/PaddleOCR
# PaddleOCR model can run on both CPU and GPU -> able to run on any machine and Colab as well
# Can be used on shfited (rotated) image
# Key Feature -> Lightweight, fast model

# 1. Install and Import Dependencies
# 1.1 GitHub Repo instalation of paddle paddle library (underlying franework)
# If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
# python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# 1.2 Install PaddleOCR Whl Package
# pip install "paddleocr>=2.0.1"
# 1.3 Clone paddle OCR repo - to get the FONTS for visualization - NOT DONE (seems to be redudant, when paddles draw_ocr function is not used)
# git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 2. Instantiate OCR Model and Detect
# Extract text from images

import config
from typing import Tuple
from cv2 import Mat
import numpy as np
from datetime import date, datetime
import os
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import nest_asyncio

nest_asyncio.apply()


# VARS FOR PADDLE OCR MODEL
ocr_model = PaddleOCR(lang=config.ocr_model_language,
                      use_angle_cls=True, show_log=False)
ocr_confidence_threshold = config.ocr_confidence_threshold

paths = config.paths


def get_timestamp_yyyy_mm_dd_hh_mm_ss() -> str:
    return (str(date.today().year) + '-' +
            str(date.today().month).zfill(2) + '-' +
            str(date.today().day).zfill(2) + '-' +
            str(datetime.now().hour) + '-' +
            str(datetime.now().minute) + '-' +
            str(datetime.now().second))


# Use the given timestamp to create a special folder for the original image,
# the original image with the detected sticky note bounding boxes
# and a crop images for every detected sticky note in the original image
def create_timestamp_folder_and_return_its_path(folder_name: str) -> str:
    recognized_images_timestamp_folder_path: str = os.path.join(
        paths['MIRO_TIMEFRAME_SNAPSHOTS'], folder_name)

    if not os.path.exists(paths['MIRO_TIMEFRAME_SNAPSHOTS']):
        os.mkdir(paths['MIRO_TIMEFRAME_SNAPSHOTS'])

    if not os.path.exists(recognized_images_timestamp_folder_path):
        os.mkdir(recognized_images_timestamp_folder_path)

    return recognized_images_timestamp_folder_path


# Saves the image img with the name img_name inside the folder with the path folder_path
def save_image_in_folder(img: np.ndarray, img_name: str, folder_path: str) -> None:
    img_path = os.path.join(folder_path, img_name)
    result: bool = cv2.imwrite(img_path, img)
    if result == True:
        print(f"File {img_name} saved successfully")
    else:
        print(f"Error in saving file {img_name}")


# TODO: DELETE IF UNUSED
debug_image_count = 0


def save_image_with_detection_for_debug(img: np.ndarray, img_file_path: str) -> None:
    if os.path.isfile(img_file_path):
        global debug_image_count
        debug_image_count = debug_image_count + 1
        original_image_with_timestamp_name: str = f"{debug_image_count}_{config.CUSTOM_MODEL_NAME_SUFFIX}_{str(config.num_steps)}.png"
        result: bool = cv2.imwrite(os.path.join(
            config.paths['MIRO_TIMEFRAME_SNAPSHOTS'], original_image_with_timestamp_name), img)

        if result == True:
            print(
                f"File {original_image_with_timestamp_name} saved successfully")
        else:
            print(
                f"Error in saving file {original_image_with_timestamp_name}")


# Saves a cropped image for every detected sticky note bounding box inside the original image
# with the detection bounding-box coordinates from img_detection_data_list
# and return a list with the temporary name of the image, the position of the bounding box and the detected color
def crop_image_to_bounding_boxes(
        img: np.ndarray,
        img_detection_data_list,
) -> list:

    cropped_images_data = []

    for index, img_detection_data in enumerate(img_detection_data_list):
        ymin = img_detection_data['ymin']
        ymax = img_detection_data['ymax']
        xmin = img_detection_data['xmin']
        xmax = img_detection_data['xmax']
        color = img_detection_data['color']

        cropped_img_according_to_its_bounding_box = img[ymin:ymax, xmin:xmax]

        cropped_img_according_to_its_bounding_box_name = f"cropped_image_{index}.png"

        cropped_images_data.append(
            {
                "position": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
                "color": color,
                "name": cropped_img_according_to_its_bounding_box_name,
                "ocr_recognized_text": "",
                "image": cropped_img_according_to_its_bounding_box
            })
    return cropped_images_data


# Euclidean distance: Given list of points on 2-D plane and an integer K (origin of the coordinate system (0,0))
# Calculate distance to K to find the closest points to the origin
# √{(x2-x1)2 + (y2-y1)2}
def euclidean(text_and_boxes_array):
    xmin = text_and_boxes_array['position']['xmin']
    ymin = text_and_boxes_array['position']['ymin']
    coor_origin_x = 0
    coor_origin_y = 0
    return ((xmin - coor_origin_x)**2 + (ymin - coor_origin_y)**2)**0.5


async def get_image_ocr_data(
    img_path,
    ocr_confidence_threshold=ocr_confidence_threshold,
) -> Tuple[list, Mat]:
    result = ocr_model.ocr(img_path)

    # convert all text into array
    # text_array = [res[1][0] for res in result]
    # print("\n \n \n")
    # for text in text_array:
    #     print(text)

    # Extracting detected components
    boxes = [res[0] for res in result]
    texts = [res[1][0] for res in result]
    scores = [res[1][1] for res in result]

    # TODO: FIND OUT IF CONVERTION IS REALLY NEEDED - MESSES UP SAVED FILES
    # By default if we import image using openCV, the image will be in BGR
    # But we want to reorder the color channel to RGB for the draw_ocd method
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    ocr_recognized_text_and_boxes_array = []

    # Visualize image and detections
    for i in range(len(result)):
        if scores[i] > ocr_confidence_threshold:
            (xmin, ymin, xmax, ymax) = (
                int(boxes[i][0][0]),
                int(boxes[i][1][1]),
                int(boxes[i][2][0]),
                int(boxes[i][3][1]))

            # position is necessary for the euclidean distance sorting of the textes
            # (and for the visualization of the bounding boxes if save_image_overlayed_with_ocr_visualization is true)
            ocr_recognized_text_and_boxes_array.append(
                {"position": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                 "text": texts[i]}
            )

    # Use euclidean distance to sort the words in the correct order from top left to bottom right
    ocr_recognized_text_and_boxes_array.sort(key=euclidean)

    return ocr_recognized_text_and_boxes_array


# Visualize ocr bounding boxes and the labels provided in ocr_recognized_text_and_boxes_array inside the image with img_path
# (optional function - only when save_image_overlayed_with_ocr_visualization flag is true)
def get_image_with_overlayed_ocr_bounding_boxes_and_text(img_path: str, ocr_recognized_text_and_boxes_array):
    img = cv2.imread(img_path)
    for ocr_recognized_text_and_box in ocr_recognized_text_and_boxes_array:
        xmin = ocr_recognized_text_and_box['position']['xmin']
        ymin = ocr_recognized_text_and_box['position']['ymin']
        xmax = ocr_recognized_text_and_box['position']['xmax']
        ymax = ocr_recognized_text_and_box['position']['ymax']
        text = ocr_recognized_text_and_box['text']

        img = cv2.rectangle(
            img=img,
            pt1=(xmin, ymin),
            pt2=(xmax, ymax),
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA)

        img = cv2.putText(
            img=img,
            text=text,
            org=(int(xmin + 5), int(ymin - 5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA)
    return img


# Remove forbidden printable ASCII characters from file name:
# < > : " / \ | ? * '
def remove_forbidden_ascii_characters(string: str) -> str:
    forbidden_characters = ['<', '>', ':',
                            '"', "\'", '/', '\\', '|', '?', '*']
    for forbidden_character in forbidden_characters:
        if forbidden_character in string:
            string = string.replace(
                forbidden_character, "")

    return string


# Create single string out of all ocr data bounding boxes
def extract_string_out_of_ocr_data(
    cropped_image: list
) -> str:
    image_ocr_text = ""
    for ocr_text_and_boxes in cropped_image['ocr_data']:
        image_ocr_text = image_ocr_text + \
            ocr_text_and_boxes['text'] + " "
    return image_ocr_text


def rename_cropped_image_with_ocr_string(cropped_image: list, index: int, image_file_path, timestamped_folder_path: str):
    new_image_file_name = f"{index}_{cropped_image['ocr_recognized_text']}"
    new_image_file_name = remove_forbidden_ascii_characters(
        new_image_file_name)
    new_image_file_path = os.path.join(
        timestamped_folder_path, f"{new_image_file_name}.png")
    os.rename(image_file_path, new_image_file_path)
    return [new_image_file_name, new_image_file_path]


# NOT IN USE!!!


# WAS USED, BUT IS CURRENTLY NOT IN USE
# http://color.lukas-stratmann.com/color-systems/hsv.html
# Get dominant color and assign it to miro sticky note color class
# sticky_note_dominant_rgb_color = get_dominant_color(new_image_file_name_with_ocr_text_in_file_name)
# sticky_note_color_class = get_sticky_note_color_class_from_rgb(sticky_note_dominant_rgb_color)

# resize image down to 1 pixel.
# def get_dominant_color(img_path):
#     img = Image.open(img_path)
#     img = img.convert("RGB")
#     img = img.resize((1, 1), resample=0)
#     dominant_rgb_color = img.getpixel((0, 0))
#     return dominant_rgb_color


# def rgb_to_hsv(r, g, b) -> Tuple[int, int, int]:
#     temp = colorsys.rgb_to_hsv(r, g, b)
#     h = int(temp[0] * 360)
#     s = int(temp[1] * 100)
#     v = round(temp[2] * 100 / 255)
#     return [h, s, v]


# def get_sticky_note_color_class_from_rgb(rgb_color):
#     [h, s, v] = rgb_to_hsv(
#         rgb_color[0],   # r
#         rgb_color[1],   # g
#         rgb_color[2]    # b
#     )

#     sticky_note_color_class = None
#     hue = h

#     if hue > 10 and hue < 60:
#         sticky_note_color_class = "yellow"
#     elif hue > 60 and hue < 180:
#         sticky_note_color_class = "green"
#     elif hue > 180 and hue < 290:
#         sticky_note_color_class = "blue"
#     elif hue > 290 or hue > 0 and hue < 10:
#         sticky_note_color_class = "red"

#     return sticky_note_color_class


def morphology_closing_operation(img):
    # Reading the input image
    # img = cv2.imread(img_path, 0)

    # Taking a matrix of size 5 as the kernel
    size = (5, 5)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    img_closed = cv2.dilate(img, kernel, iterations=1)
    img_closed = cv2.erode(img, kernel, iterations=1)

    return img_closed


def morphology_opening_operation(img):
    # Reading the input image
    # img = cv2.imread(img_path, 0)

    # Taking a matrix of size 5 as the kernel
    size = (5, 5)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    img_opened = cv2.erode(img, kernel, iterations=1)
    img_opened = cv2.dilate(img, kernel, iterations=1)

    return img_opened


# def opencv_script_thresholding(img_path):
#     img = cv2.imread(img_path, 0)
#     # global thresholding
#     ret1, th1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
#     # Otsu's thresholding
#     ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     # Otsu's thresholding after Gaussian filtering
#     blur = cv2.GaussianBlur(img, (5, 5), 0)
#     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     # plot all the images and their histograms
#     images = [img, 0, th1,
#               img, 0, th2,
#               blur, 0, th3]
#     titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
#               'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
#               'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
#     for i in range(3):
#         plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
#         plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#         plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
#         plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#         plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
#         plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#     plt.show()

# https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/


def opencv_script_thresholding(img_path):
    img = cv2.imread(img_path, 0)

    # Global thresholding
    ret1, global_threshold_img = cv2.threshold(
        img, 100, 255, cv2.THRESH_BINARY)

    # Adaptive thresholding
    adaptive_threshold_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

    # Adaptive thresholding after Gaussian filtering
    adaptive_threshold_gaussian_blur_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                 cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding
    ret4, otus_threshold_img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret5, otus_threshold_gaussian_blur_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, global_threshold_img,
              img, 0, adaptive_threshold_img,
              blur, 0, adaptive_threshold_gaussian_blur_img,
              img, 0, otus_threshold_img,
              blur, 0, otus_threshold_gaussian_blur_img]

    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', 'Adaptive thresholding',
              'Gaussian filtered Image', 'Histogram', 'Adaptive thresholding after Gaussian filtering',
              'Original Noisy Image', 'Histogram', 'Otsus Thresholding',
              'Gaussian filtered Image', 'Histogram', 'Otsus Thresholding after Gaussian filtering']

    for i in range(5):
        plt.subplot(5, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(5, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

        plt.subplot(5, 3, i*3+3), plt.imshow((images[i*3+2]), 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


# # TEST WITH PYTESSERACT
# # !pip install pytesseract
# # install tesseract via tesseract-ocr-w64-setup-v5.2.0.20220712.exe (64 bit) on https://github.com/UB-Mannheim/tesseract/wiki
# # reference path
# # Ctrl+CLick on pytesseract to see all functions AND Ctrl+Click on function to see function definition
# # myconfig = r"--psm 11 --oem 3"
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# sticky_notes_test_images_path = 'C:\_WORK\\GitHub\_data-science\TFODCourse\sticky_notes_test_images'

# img = cv2.imread(os.path.join(
#     sticky_notes_test_images_path, "medium.jpg"))

# height, width, _ = img.shape

# # WARNING: pytesseract library only accept RGB values and openCV is in BGR
# # Therefore, convertion from BGR to RGB is needed
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # image_to_string
# # Returns result of a Tesseract OCR run on provided image to string
# # print(pytesseract.image_to_string(img))

# data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)
# amount_of_boxes = len(data['text'])

# for i in range(amount_of_boxes):
#     if float(data['conf'][i]) > 80:
#         (x, y, width, height) = (
#             data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#         img = cv2.rectangle(
#             img, (x, y), (x + width, y + height), (0, 255, 0), 2)
#         img = cv2.putText(img, data['text'][i], (x, y + height+20),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.cv2.LINE_AA)

# cv2.imshow('img', img)
# cv2.waitKey(0)
