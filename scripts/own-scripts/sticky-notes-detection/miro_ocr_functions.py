

from dataclasses import dataclass
import numpy as np
from PIL import Image
from datetime import date, datetime
import os
from random import random
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import asyncio
import nest_asyncio
import config
nest_asyncio.apply()


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


# VARS FOR PADDLE OCR MODEL
img_path = 'C:\_WORK\\GitHub\_data-science\TFODCourse\sticky_notes_test_images\single_difficult.jpg'
ocr_model = PaddleOCR(lang='german')
ocr_confidence_threshold = 0.50

paths = config.paths


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


def get_timestamp_yyyy_mm_dd_hh_mm_ss() -> str:
    return (str(date.today().year) + '-' +
            str(date.today().month).zfill(2) + '-' +
            str(date.today().day).zfill(2) + '-' +
            str(datetime.now().hour) + '-' +
            str(datetime.now().minute) + '-' +
            str(datetime.now().second))


def create_timestamp_folder_and_return_its_path(timestamp: str) -> str:
    # TODO: Create folder path in config specifically for those timestamp backups
    recognized_images_timestamp_folder_path: str = os.path.join(
        paths['MIRO_TIMEFRAME_SNAPSHOTS'], timestamp)
    # f"C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\IMAGE_DATA_STICKY_NOTES\\{timestamp}"

    # TODO: FIND BETTER FOLDER STRUCTURE LOGIC
    if not os.path.exists(paths['MIRO_TIMEFRAME_SNAPSHOTS']):
        os.mkdir(paths['MIRO_TIMEFRAME_SNAPSHOTS'])

    if not os.path.exists(recognized_images_timestamp_folder_path):
        os.mkdir(recognized_images_timestamp_folder_path)

    return recognized_images_timestamp_folder_path


def save_image_with_timestamp(img: np.ndarray, img_file_path: str, timestamp: str, timestamped_folder_path: str, suffix="") -> None:
    if os.path.isfile(img_file_path):
        original_image_with_timestamp_name: str = f"{timestamp}{suffix}.png"
        result: bool = cv2.imwrite(os.path.join(
            timestamped_folder_path, original_image_with_timestamp_name), img)

        if result == True:
            print(
                f"File {original_image_with_timestamp_name} saved successfully")
        else:
            print(
                f"Error in saving file {original_image_with_timestamp_name}")


# def get_dominant_color(pil_img):
#     img = pil_img.copy()
#     img = img.convert("RGBA")
#     img = img.resize((1, 1), resample=0)
#     dominant_color = img.getpixel((0, 0))
#     return dominant_color


def crop_and_save_recognized_images(
        img: np.ndarray,
        img_detection_bounding_boxes,
        timestamped_folder_path: str):

    cropped_images_data = []

    for index, img_detection_bounding_box in enumerate(img_detection_bounding_boxes):
        ymin = img_detection_bounding_box['ymin']
        ymax = img_detection_bounding_box['ymax']
        xmin = img_detection_bounding_box['xmin']
        xmax = img_detection_bounding_box['xmax']
        color = img_detection_bounding_box['color']

        cropped_img_according_to_its_bounding_box = img[ymin:ymax, xmin:xmax]

        # img_file_path = "C:\\Users\\vbraz\\Desktop\\IMAGE_DATA_STICKY_NOTES\\randy-bachelor-sticky-notes-images\\IMG_0270.JPG"
        # pilImg = Image.open(img_file_path)
        # cropped_img_according_to_its_bounding_box = pilImg.crop(
        #     (ymin, ymax, xmin, xmax))
        # dominent_color = get_dominant_color(
        #     cropped_img_according_to_its_bounding_box)
        # print(dominent_color)

        # data = np.reshape(cropped_img_according_to_its_bounding_box, (-1, 3))
        # print(data.shape)
        # data = np.float32(data)
        # criteria = (cv2.TERM_CRITERIA_EPS +
        #             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # flags = cv2.KMEANS_RANDOM_CENTERS
        # compactness, labels, centers = cv2.kmeans(
        #     data, 1, None, criteria, 10, flags)

        # print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))

        cropped_img_according_to_its_bounding_box_name = f"cropped_image_{index}.png"

        cropped_images_data.append(
            {
                "position": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
                "color": color,
                "name": cropped_img_according_to_its_bounding_box_name,
                "ocr_recognized_text": "",
            })

        result = cv2.imwrite(os.path.join(
            timestamped_folder_path, cropped_img_according_to_its_bounding_box_name), cropped_img_according_to_its_bounding_box)

        if result == True:
            print(
                f"File {cropped_img_according_to_its_bounding_box_name} saved successfully")
        else:
            print(
                f"Error in saving file {cropped_img_according_to_its_bounding_box_name}")

    return cropped_images_data


def remove_forbidden_ascii_characters(string):
    forbidden_characters = ['<', '>', ':',
                            '"', "\'", '/', '\\', '|', '?', '*']
    for forbidden_character in forbidden_characters:
        if forbidden_character in string:
            string = string.replace(
                forbidden_character, "")

    return string


async def get_image_text_data_by_ocr_for_each_file_in_timestamped_folder_and_save_it(cropped_images_data: list, timestamped_folder_path: str, ocr_confidence_threshold=0.5, visualize_text_in_image=False):
    for index, cropped_image_data in enumerate(cropped_images_data):
        image_file_path = os.path.join(
            timestamped_folder_path, cropped_image_data['name'])
        # checking if it is a file
        if os.path.isfile(image_file_path):
            image_ocr_data_array = await asyncio.create_task(get_image_text_data_by_ocr(image_file_path, ocr_confidence_threshold, visualize_text_in_image=visualize_text_in_image))
            image_with_ocr_data_visualization = image_ocr_data_array[1]
            # TODO: Ensure that order of words is correct -> maybe sort after x-y index
            # TODO: Find out why color of the saved file is still black and not white
            # TODO: Find out, why color does not change
            cv2.imwrite(image_file_path, image_with_ocr_data_visualization)

            # get every ocr bounding box and create a string collecting all ocr bounding boxes
            image_ocr_text = ""
            for image_ocr_data in image_ocr_data_array[0]:
                image_ocr_text = image_ocr_text + \
                    " " + image_ocr_data['text']

            new_image_file_name = f"{index}_{image_ocr_text}.png"
            print(new_image_file_name)

            # remove forbidden printable ASCII characters from file name:
            # < > : " / \ | ? * '
            new_image_file_name = remove_forbidden_ascii_characters(
                new_image_file_name)

            print(new_image_file_name)
            new_image_file_name_with_ocr_information = os.path.join(
                timestamped_folder_path, new_image_file_name)
            print(image_file_path)
            print(new_image_file_name_with_ocr_information)
            os.rename(image_file_path, new_image_file_name_with_ocr_information)

            # override name with the new one
            cropped_image_data['name'] = image_ocr_text
            cropped_image_data['ocr_recognized_text'] = image_ocr_text

    if visualize_text_in_image:
        cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

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


async def get_image_text_data_by_ocr(img_path, ocr_confidence_threshold=ocr_confidence_threshold, visualize_text_in_image=True):
    result = ocr_model.ocr(img_path)

    # convert all text into array
    # text_array = [res[1][0] for res in result]
    # print("\n \n \n")
    # for text in text_array:
    #     print(text)

    # 3. Visualise Results
    # Extracting detected components
    boxes = [res[0] for res in result]
    texts = [res[1][0] for res in result]
    scores = [res[1][1] for res in result]

    # Import image
    img = cv2.imread(img_path)

    # By default if we import image using openCV, the image will be in BGR
    # But we want to reorder the color channel to RGB for the draw_ocd method
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    text_and_boxes_array = []

    # Visualize image and detections
    for i in range(len(result)):
        if scores[i] > ocr_confidence_threshold:
            (xmin, ymin, xmax, ymax) = (
                int(boxes[i][0][0]),
                int(boxes[i][1][1]),
                int(boxes[i][2][0]),
                int(boxes[i][3][1]))

            text_and_boxes_array.append(
                {"position": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
                 "text": texts[i]}
            )

            if visualize_text_in_image:
                img = cv2.rectangle(
                    img=img,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA)

                img = cv2.putText(
                    img=img,
                    text=texts[i],
                    org=(int(xmin + 5), int(ymin - 5)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)

    # Ensure that order of words is correct by sorting  after their euclidean distance
    text_and_boxes_array.sort(key=euclidean)

    if visualize_text_in_image:
        # cv2.namedWindow(f"OCR of {img_path}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"OCR of {random()}", img)

    return [text_and_boxes_array, img]


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
