# pip install requests
# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro-sticky-notes-sync.py

import nest_asyncio
import asyncio
import aiohttp
import requests
import tensorflow as tf
import os
from paddleocr import PaddleOCR
from cgitb import text
import matplotlib.pyplot as plt
import cv2
import numpy as np

# from object_detection.utils import config_util
# from object_detection.builders import model_builder
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import label_map_util

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


files = config.files
paths = config.paths

# VARS FOR MIRO REST API
DEBUG_PRINT_RESPONSES = False
auth_token = "eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_Bw1UPxNElmWbBGy8MfSWWJWOCLs"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {auth_token}"
}

# VARS FOR PADDLE OCR MODEL
img_path = 'C:\_WORK\\GitHub\_data-science\TFODCourse\sticky_notes_test_images\single_difficult.jpg'
ocr_model = PaddleOCR(lang='german')
ocr_confidence_threshold = 0.50


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


async def get_image_text_data_by_ocr(img_path, ocr_confidence_threshold: float, visualize_text_in_image: bool):
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
                {"positions": [xmin, ymin, xmax, ymax], "text": texts[i]})

            if visualize_text_in_image:
                img = cv2.rectangle(
                    img=img,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA)

                img = cv2.putText(
                    img=img,
                    text=texts[i],
                    org=(int(xmin + 5), int(ymin + 20)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA)

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', img)
    cv2.waitKey(0)
    return text_and_boxes_array


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


opencv_script_thresholding(
    "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).jpeg")


# VARS FOR REAL TIME OBJECT DETECTION
tf.config.run_functions_eagerly(True)
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(
    files['CUSTOM_MODEL_PIPELINE_CONFIG'])
detection_model = model_builder.build(
    model_config=configs['model'], is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(
    files['LABELMAP'])


# MIRO REST API HELPER FUNCTIONS
# GET ALL BOARD IDS AND NAMES
# WARNING:
# seems that the Get boards REST API function is not working properly
# sometimes it only returns one board, even if there are more
async def get_all_miro_board_names_and_ids():
    global global_session
    url = "https://api.miro.com/v2/boards?limit=50&sort=default"
    async with global_session.get(url, headers=headers) as resp:
        response = await resp.json()

        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return [{"name": board['name'], "id": board['id']} for board in response['data']]

# GET ALL BOARD ITEMS


async def get_all_items():
    global global_session
    global global_board_id

    get_all_items_limit = 50
    get_all_items_type = "sticky_note"

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D' )}/items?limit={get_all_items_limit}&type={get_all_items_type}"

    async with global_session.get(url, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())
        return response['data']

# DELETE BOARD ITEM


async def delete_item(item_id):
    global global_board_id
    global global_session
    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/items/{item_id}"

    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_Bw1UPxNElmWbBGy8MfSWWJWOCLs"
    }

    response = requests.delete(url, headers=headers)
    if DEBUG_PRINT_RESPONSES:
        print(await response.text())

    # WARNING: Seems not to work with global_session.delete(url, headers=headers)
    #     async with global_session.delete(url, headers=headers) as resp:
    #         response = await resp.json()
    #         if DEBUG_PRINT_RESPONSES: print(await resp.text())

# DELETE ALL BOARD ITEMS


async def delete_all_items():
    global global_session
    board_items = await asyncio.create_task(get_all_items())
    for board_item in board_items:
        await asyncio.create_task(delete_item(board_item['id']))

# CREATE ITEM


async def create_item(item_position):
    global global_session
    global global_board_id

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/sticky_notes"

    payload = {
        "data": {"shape": "square"},
        "position": {
            "origin": "center",
            "x": item_position['xmin'],
            "y": item_position['ymin']
        },
        "geometry": {
            #             "height": item_position['ymax'] - item_position['ymin'],
            "width": item_position['xmax'] - item_position['xmin']
        }
    }

    async with global_session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

# CREATE LIST OF ITEMS


async def create_all_items(item_positions):
    global global_session
    global global_board_id

    url = f"https://api.miro.com/v2/boards/{global_board_id.replace('=', '%3D')}/sticky_notes"

    for item_position in item_positions:
        payload = {
            "data": {"shape": "square"},
            "position": {
                "origin": "center",
                "x": item_position['xmin'],
                "y": item_position['ymin']
            },
            "geometry": {
                #             "height": item_position['ymax'] - item_position['ymin'],
                "width": item_position['xmax'] - item_position['xmin']
            }
        }

        async with global_session.post(url, json=payload, headers=headers) as resp:
            response = await resp.json()
            if DEBUG_PRINT_RESPONSES:
                print(await resp.text())
            print(await resp.text())


# FUNCTIONS NECESSARY FOR REAL TIME OBJECT DETECTION
@ tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def get_detections_from_img(image):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    return detections


def visualize_detections_from_image(
    image,
    detections,
    category_index,
    min_score_thresh: float,
    line_thickness: int,
    print_results=True
):
    image_np = np.array(image)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=50,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        line_thickness=line_thickness)

    if print_results:
        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()

    return image_np_with_detections

# LOAD THE LATEST CHECKPOINT OF THE OBJECT DETECTION MODEL
# Seems to be necessary for any visual detection


def load_latest_checkpoint_of_custom_object_detection_model():
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(
        paths['CUSTOM_MODEL_PATH'], 'ckpt-1')).expect_partial()


async def scan_for_object_in_video(print_results: bool):
    # ValueError: 'images' must have either 3 or 4 dimensions. -> could be related to wrong source of VideoCapture!
    cap = cv2.VideoCapture(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scan_condition = cap.isOpened()
    storaged_scanned_object_detection_boxes = []

    while scan_condition:
        # await asyncio.sleep(1)
        ret, frame = cap.read()
        frame_detections = get_detections_from_img(frame)
        frame_detections_np_with_detections = visualize_detections_from_image(
            frame,
            frame_detections,
            category_index,
            min_score_thresh=0.8,
            line_thickness=10,
            print_results=print_results
        )

        cv2.imshow('object detection',  cv2.resize(
            frame_detections_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        # filter data for only the necessary information
        formatted_scanned_object_detection_boxes = []
        for index, frame_detection_score in enumerate(frame_detections['detection_scores'][0]):
            if frame_detection_score > 0.8:
                # same as in "11. Auto-Labeling" of "2. Training and Detection"
                scanned_object_detection_box = frame_detections['detection_boxes'][0][index]
                ymin = round(float(scanned_object_detection_box[0] * height))
                xmin = round(float(scanned_object_detection_box[1] * width))
                ymax = round(float(scanned_object_detection_box[2] * height))
                xmax = round(float(scanned_object_detection_box[3] * width))
                formated_scanned_object_detection_box = {
                    "ymin": ymin,
                    "xmin": xmin,
                    "ymax": ymax,
                    "xmax": xmax,
                    #                     "timestamp": datetime.timestamp(datetime.now())
                }
#                 print(f"- bounding boxes (relative): {scanned_object_data} \n")

                formatted_scanned_object_detection_boxes.append(
                    formated_scanned_object_detection_box)

        print(
            f"formatted_scanned_object_detection_boxes {formatted_scanned_object_detection_boxes}")

#         if len(formatted_scanned_object_detection_boxes) > len(storaged_scanned_object_detection_boxes):
#             storaged_scanned_object_detection_boxes = formatted_scanned_object_detection_boxes.copy()

#         print("CHECKING NEW SCAN")
#         # check local storage
#         for formatted_scanned_object_detection_box in formatted_scanned_object_detection_boxes:
#             for storaged_scanned_object_detection_box in storaged_scanned_object_detection_boxes:
#                 print(f"formated_scanned_object_detection_box['ymin']: {formated_scanned_object_detection_box['ymin']}")
#                 print(f"storaged_scanned_object_detection_box['ymin']: {storaged_scanned_object_detection_box['ymin']}")

#                 # find object which already exist in storage
#                 if formated_scanned_object_detection_box['ymin'] > storaged_scanned_object_detection_box['ymin'] - 5 and formated_scanned_object_detection_box['ymin'] < storaged_scanned_object_detection_box['ymin'] + 5:
#                     print(f"Object with the coordinates x: {formated_scanned_object_detection_box['xmin']} and y: {formated_scanned_object_detection_box['ymin']} already exist.")

#                     # Find out if object has been moved
#                     print(storaged_scanned_object_detection_box['ymin'] - formated_scanned_object_detection_box['ymin'])

#         print(f"storaged_scanned_object_detection_boxes count: {len(storaged_scanned_object_detection_boxes)}")
#         print(f"storaged_scanned_object_detection_boxes: {storaged_scanned_object_detection_boxes}")


#         # get all miro board items
        board_items = await asyncio.create_task(get_all_items())

        # compare miro board item count with real world item count
        # add missing items to miro board
        board_items_count = len(board_items)
        last_index = len(formatted_scanned_object_detection_boxes)
        print(f"Identify {board_items_count} miro sticky note widgets from the scanned {len(formatted_scanned_object_detection_boxes)} sticky notes in real world")

        await asyncio.create_task(delete_all_items())
        while board_items_count < len(formatted_scanned_object_detection_boxes):
            if len(formatted_scanned_object_detection_boxes) > 0:
                await asyncio.create_task(create_item(formatted_scanned_object_detection_boxes[last_index - 1]))
                last_index = last_index - 1
                board_items_count = board_items_count + 1


#                 for data in scanned_object_data_list:
#                     if scanned_object_data['ymin'] > data['ymin'] - 5 and scanned_object_data['ymin'] < data['ymin'] + 5:
#                         print("TRUE")

# #                 if scanned_object_data in scanned_object_data_list:
# #                     print(f"Object {scanned_object_data} already exist.")
# #                 else:
# #                     scanned_object_data_list.append(scanned_object_data)
# #                     await asyncio.create_task(create_item(scanned_object_data))
# #                     print(f"Added scanned_object_data {scanned_object_data} to the list and created it on miro board.")


#         print(scanned_object_data_list)
#         # store scanned_object_positions, if not already stores
#         # detect changes in scanned_object_positions
#         # Get all items on board
#         # Check if some items exist in real-world, which are missing on miro board
# #         await asyncio.create_task(create_all_items(scanned_object_data_list))

    return storaged_scanned_object_detection_boxes


# Create new Miro-Board if no Board with given name exists, else return the id or the existing one
async def create_new_miro_board_or_get_existing(name: str, description: str):
    global global_session

    board_names_and_ids = await asyncio.create_task(get_all_miro_board_names_and_ids())

    for board_name_and_id in board_names_and_ids:
        if board_name_and_id['name'] == name:
            print(f"WARNING: The board with the name {name} already exist. \n")
            return board_name_and_id['id']

    url = "https://api.miro.com/v2/boards"

    payload = {
        "name": name,
        "description": description,
        "policy": {
            "permissionsPolicy": {
                "collaborationToolsStartAccess": "all_editors",
                "copyAccess": "anyone",
                "sharingAccess": "team_members_with_editing_rights"
            },
            "sharingPolicy": {
                "access": "private",
                "inviteToAccountAndBoardLinkAccess": "no_access",
                "organizationAccess": "private",
                "teamAccess": "private"
            }
        }
    }

    async with global_session.post(url, json=payload, headers=headers) as resp:
        response = await resp.json()
        if DEBUG_PRINT_RESPONSES:
            print(await resp.text())

    return response['id']


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


# Resizing display area

# plt.figure(figsize=(15, 15))
# font = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin')
# complete_path = os.path.join(
#     'C:\_WORK\GitHub\_data-science\TFODCourse\Tensorflow\scripts\PaddleOCR\doc\fonts\latin')
# default_font = ImageFont.load_default()
# font_path_2 = os.path.join(sticky_notes_test_images_path, "german")
# font = ImageFont.truetype("arial.ttf", 15)

# print(complete_path)
# annotated = draw_ocr(img, boxes, texts, scores, font_path=complete_path)
# plt.imshow(annotated)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# MAIN FUNCTION
global_session = None
global_board_id = None


async def main():
    board_name = "STICKY NOTES SYNC BOARD"
    board_description = "Board to test the synchronization of the sticky notes in the real life and the here created miro board."

    async with aiohttp.ClientSession() as session:
        global global_session
        global global_board_id
        global_session = session

        # board_id = await asyncio.create_task(create_new_miro_board_or_get_existing(board_name, board_description))
        # print(f"board_id: {board_id}")
        # global_board_id = board_id

#         board_items = await asyncio.create_task(get_all_items(board_id))
#         print(f"board_items: {board_items}")

        # load_latest_checkpoint_of_custom_object_detection_model()

        # RuntimeError: (PreconditionNotMet) The third-party dynamic library (cudnn64_7.dll) that Paddle depends on is not configured correctly. (error code is 126)

        # threshold_img_with_global_threshold(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # threshold_img_with_otsu_binarization(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # threshold_img_with_otsu_binarization_and_gaussian_blue(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        opencv_script_thresholding(
            "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # await asyncio.create_task(get_image_text_data_by_ocr("C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png", ocr_confidence_threshold, True))

# await asyncio.create_task(scan_for_object_in_video(
#     print_results=False,
# ))


asyncio.run(main())
