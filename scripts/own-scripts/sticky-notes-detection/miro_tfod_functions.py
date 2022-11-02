
# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro-sticky-notes-sync.py

import config
import nest_asyncio
import asyncio
import aiohttp
from miro_rest_api_functions import \
    get_all_miro_board_names_and_ids, \
    get_all_items, delete_item, \
    delete_all_items, \
    create_sticky_note, \
    create_all_sticky_notes, \
    create_new_miro_board_or_get_existing
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from os.path import isfile, join
from os import listdir
import re

nest_asyncio.apply()


files = config.files
paths = config.paths


# VARS FOR REAL TIME OBJECT DETECTION
tf.config.run_functions_eagerly(True)
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(
    files['CUSTOM_MODEL_PIPELINE_CONFIG'])
detection_model = model_builder.build(
    model_config=configs['model'], is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(
    files['LABELMAP'])

min_score_thresh = config.min_score_thresh
bounding_box_and_label_line_thickness = config.bounding_box_and_label_line_thickness


# LOAD THE LATEST CHECKPOINT OF THE OBJECT DETECTION MODEL
# Necessary for any visual detection
def get_maximum_trained_model_checkpoint():
    maximum_ckpt_number = -1
    for file in listdir(paths['CUSTOM_MODEL_PATH']):
        try:
            file_ckpt_number = int(
                re.search(r'ckpt-(.*).index', file).group(1))
            if maximum_ckpt_number == -1:
                maximum_ckpt_number = file_ckpt_number
            if maximum_ckpt_number < file_ckpt_number:
                maximum_ckpt_number = file_ckpt_number
        except AttributeError:
            continue
    return maximum_ckpt_number


# Restore the latest checkpoint of trained model
# The checkpoint files correspond to snapshots of the model at given steps
# The latest checkpoint is the most trained state of the model
def load_latest_checkpoint_of_custom_object_detection_model():
    maximum_ckpt_number = get_maximum_trained_model_checkpoint()
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(
        paths['CUSTOM_MODEL_PATH'], f"ckpt-{maximum_ckpt_number}")).expect_partial()


# FUNCTIONS NECESSARY FOR REAL TIME OBJECT DETECTION
@ tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# use the detection model to preprocess, predict and postprocess image to get detections
def get_detections_from_img(image):
    image_np = np.array(image)
    # converting image into tensor object
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    # make detections on image
    detections = detect_fn(input_tensor)
    return detections


# returns a list with the relative ymin, xmin, ymax, xmax coordinates of those boundingboxes, which are above the min_score_thresh
def get_bounding_boxes_above_min_score_thresh(detections, imgHeight, imgWidth, min_score_thresh):
    formatted_scanned_object_detection_boxes = []
    for index, detection_score in enumerate(detections['detection_scores'][0]):
        if detection_score > min_score_thresh:
            scanned_object_detection_box = detections['detection_boxes'][0][index]
            # scanned_object_class = int(
            #     detections['detection_classes'][0][index] + 1)

            # values of bounding-boxes must be treated as relative coordinates to the image instead as absolute
            ymin = round(float(scanned_object_detection_box[0] * imgHeight))
            xmin = round(float(scanned_object_detection_box[1] * imgWidth))
            ymax = round(float(scanned_object_detection_box[2] * imgHeight))
            xmax = round(float(scanned_object_detection_box[3] * imgWidth))

            # formated_scanned_object_label = list(
            #     filter(lambda label: (
            #         label['id'] == scanned_object_class), config.labels)
            # )

            formated_scanned_object_detection_box = {
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                # "color": formated_scanned_object_label[0]['color']
            }

            # print(f"- bounding boxes (relative): {scanned_object_data} \n")

            formatted_scanned_object_detection_boxes.append(
                formated_scanned_object_detection_box)

    return formatted_scanned_object_detection_boxes


# returns the given image with its overlay labeled bounding boxes from the passed detections (formatted with scores and label names)
def get_image_with_overlayed_labeled_bounding_boxes(
    image,
    detections,
    category_index=category_index,
    min_score_thresh=min_score_thresh,
    line_thickness=bounding_box_and_label_line_thickness,
    visualize_overlayed_labeled_bounding_boxes_in_image=True
):
    image_np = np.array(image)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    # necessary for detections['detection_classes']
    # because label ids start with 1 and detections['detection_classes'] start with 0
    label_id_offset = 1
    # make copy, because visualize_boxes_and_labels_on_image_array modifies image in place
    image_np_with_detections = image_np.copy()

    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
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

    if visualize_overlayed_labeled_bounding_boxes_in_image:
        # TODO: plt.show seems not to work, replace it with cv2.imshow?
        # plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        # plt.show()
        cv2.imshow("Visualize bounding-box detections in image",
                   image_np_with_detections)
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    return image_np_with_detections


# NOT IN USE!!!
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
        frame_detections_np_with_detections = get_image_with_overlayed_labeled_bounding_boxes(
            frame,
            frame_detections,
            category_index,
            min_score_thresh=min_score_thresh,
            line_thickness=bounding_box_and_label_line_thickness,
            print_results=print_results
        )

        cv2.imshow('object detection',  cv2.resize(
            frame_detections_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        formatted_scanned_object_detection_boxes = get_bounding_boxes_above_min_score_thresh(
            detections=frame_detections, imgHeight=height, imgWidth=width)

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
                await asyncio.create_task(create_sticky_note(formatted_scanned_object_detection_boxes[last_index - 1]))
                last_index = last_index - 1
                board_items_count = board_items_count + 1


#                 for data in scanned_object_data_list:
#                     if scanned_object_data['ymin'] > data['ymin'] - 5 and scanned_object_data['ymin'] < data['ymin'] + 5:
#                         print("TRUE")

# #                 if scanned_object_data in scanned_object_data_list:
# #                     print(f"Object {scanned_object_data} already exist.")
# #                 else:
# #                     scanned_object_data_list.append(scanned_object_data)
# #                     await asyncio.create_task(create_sticky_note(scanned_object_data))
# #                     print(f"Added scanned_object_data {scanned_object_data} to the list and created it on miro board.")


#         print(scanned_object_data_list)
#         # store scanned_object_positions, if not already stores
#         # detect changes in scanned_object_positions
#         # Get all items on board
#         # Check if some items exist in real-world, which are missing on miro board
# #         await asyncio.create_task(create_all_sticky_notes(scanned_object_data_list))

    return storaged_scanned_object_detection_boxes
