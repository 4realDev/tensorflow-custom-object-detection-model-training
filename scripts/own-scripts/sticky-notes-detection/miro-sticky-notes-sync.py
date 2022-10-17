
# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro-sticky-notes-sync.py

import os
from datetime import date, datetime
from genericpath import exists
import numpy as np
import cv2
import tensorflow as tf

import config

from miro_rest_api_functions import \
    get_all_miro_board_names_and_ids, \
    get_all_items, \
    delete_item, \
    delete_all_items, \
    create_item, \
    create_frame, \
    create_image, \
    create_all_items, \
    create_new_miro_board_or_get_existing

from miro_tfod_functions import \
    visualize_detections_from_image, \
    get_detections_from_img, \
    get_bounding_boxes_above_min_score_thres, \
    load_latest_checkpoint_of_custom_object_detection_model, \
    scan_for_object_in_video

from miro_ocr_functions import \
    get_image_text_data_by_ocr, \
    save_image_with_timestamp, \
    get_timestamp_yyyy_mm_dd_hh_mm_ss, \
    create_timestamp_folder_and_return_its_path, \
    crop_and_save_recognized_images, \
    get_image_text_data_by_ocr_for_each_file_in_timestamped_folder_and_save_it

import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()

files = config.files
paths = config.paths

global_session = None
global_board_id = None


async def backup_recognized_image_data(img_file_path):
    timestamp = get_timestamp_yyyy_mm_dd_hh_mm_ss()
    timestamped_folder_path = create_timestamp_folder_and_return_its_path(
        timestamp)
    img = cv2.imread(img_file_path)
    img_detections = get_detections_from_img(img)
    img_detection_bounding_boxes = get_bounding_boxes_above_min_score_thres(
        detections=img_detections, imgHeight=img.shape[0], imgWidth=img.shape[1])
    img_with_detection_bounding_boxes = visualize_detections_from_image(
        img, img_detections, visualize_bounding_box_detections_in_image=True)

    save_image_with_timestamp(
        img, img_file_path, timestamp, timestamped_folder_path, suffix="-original")

    save_image_with_timestamp(
        img_with_detection_bounding_boxes, img_file_path, timestamp, timestamped_folder_path, suffix="-with_bounding_boxes")

    cropped_images_data = crop_and_save_recognized_images(
        img, img_detection_bounding_boxes, timestamped_folder_path)

    cropped_images_data_with_ocr_text = await asyncio.create_task(get_image_text_data_by_ocr_for_each_file_in_timestamped_folder_and_save_it(cropped_images_data, timestamped_folder_path, visualize_text_in_image=False))

    return [cropped_images_data_with_ocr_text, timestamp]


# def is_frame_with_todays_timestamp_existing(frames_data, timestamp):
#     frame_exists = False
#     frames_data_set = set(frames_data['data'])
#     for i, frame_data in enumerate(frames_data_set):
#         print(frame_data['createdAt'].split('T')[0])
#         # data{createdAt:"2022-04-21T07:15:54Z"}
#         if timestamp in frame_data['createdAt'].split('T')[0]:
#             frame_exists = True
#     return frame_exists


async def main():
    async with aiohttp.ClientSession() as session:
        global global_session
        global global_board_id
        global_session = session

        # board_id = await asyncio.create_task(create_new_miro_board_or_get_existing(board_name, board_description))
        # print(f"board_id: {cropped_imageboard_id}")
        # global_board_id = board_id

#         board_items = await asyncio.create_task(get_all_items(board_id))
#         print(f"board_items: {board_items}")

        load_latest_checkpoint_of_custom_object_detection_model()

        img_file_path = "C:\\Users\\vbraz\\Desktop\\IMAGE_DATA_STICKY_NOTES\\randy-bachelor-sticky-notes-images\\IMG_0264.JPG"
        try:
            with open(img_file_path) as f:
                print("File present")
        except FileNotFoundError:
            print('\n\n\nFile is not present\n\n\n')

        [sticky_notes_data, timestamp] = await asyncio.create_task(backup_recognized_image_data(img_file_path))

        board_id = await asyncio.create_task(create_new_miro_board_or_get_existing(name=timestamp, description=timestamp, session=session))
        print(f"board_id: {board_id}")
        # global_board_id = board_id
        # for sticky_note_data in sticky_notes_data:
        #     print(sticky_note_data['position'])
        #     print(sticky_note_data['ocr_recognized_text'])

        # TODO: Calculate average size of sticky note and set it as width and height (?)
        # What is some is wrong recognized and influences every other - otherwise stickynotes should have equal height and width
        for sticky_note_data in sticky_notes_data:
            await asyncio.create_task(create_item(
                sticky_note_data['position'],
                sticky_note_data['color'],
                sticky_note_data['ocr_recognized_text'],
                board_id,
                session)
            )

        # min_x_of_sticky_notes_data = min(sticky_notes_data['position']['xmin'])
        # min_y_of_sticky_notes_data = min(sticky_notes_data['position']['ymin'])
        # max_x_of_sticky_notes_data = min(sticky_notes_data['position']['xmax'])
        # max_y_of_sticky_notes_data = min(sticky_notes_data['position']['ymax'])
        # frame_height = max_y_of_sticky_notes_data - min_y_of_sticky_notes_data
        # frame_width = max_x_of_sticky_notes_data - min_x_of_sticky_notes_data
        # frames_data = await asyncio.create_task(get_all_items("frame"))
        # frame_exists = is_frame_with_todays_timestamp_existing(
        #     frames_data, timestamp)

        #   "data": [
        #     {
        #       "id": "3458764535167760899",
        #       "type": "frame",
        #       "links": {
        #         "self": "https://api.miro.com/v2/boards/uXjVPQw7YWQ=/frames/3458764535167760899",
        #         "related": "https://api.miro.com/v2/boards/uXjVPQw7YWQ=/items?parent_item_id=3458764535167760899&limit=10&cursor="
        #       },
        #       "createdAt": "2022-10-05T10:31:02Z",
        #       "createdBy": {
        #         "id": "3458764516565255692",
        #         "type": "user"
        #       },
        #       "data": {
        #         "format": "custom",
        #         "title": "Sample frame title",
        #         "type": "freeform"
        #       },
        #       "geometry": {
        #         "width": 896,
        #         "height": 508
        #       },
        #       "modifiedAt": "2022-10-05T10:32:24Z",
        #       "modifiedBy": {
        #         "id": "3458764516565255692",
        #         "type": "user"
        #       },
        #       "position": {
        #         "x": -120.403183796224,
        #         "y": 274.361353240576,
        #         "origin": "center",
        #         "relativeTo": "canvas_center"
        #       }
        #     },

        # await asyncio.create_task(create_frame(0, 0, str(timestamp), frame_height, frame_width, board_id, session))

        # visualize_detections_from_image(img, img_detections)

        # RuntimeError: (PreconditionNotMet) The third-party dynamic library (cudnn64_7.dll) that Paddle depends on is not configured correctly. (error code is 126)
        # TODO: Try to run Paddle OCR on GPU and uninstall CPU package -> pip uninstall paddlepaddle -i https://mirror.baidu.com/pypi/simple

        # threshold_img_with_global_threshold(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # threshold_img_with_otsu_binarization(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # threshold_img_with_otsu_binarization_and_gaussian_blue(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # opencv_script_thresholding(
        #     "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png")
        # await asyncio.create_task(get_image_text_data_by_ocr("C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png", ocr_confidence_threshold, True))

        # await asyncio.create_task(scan_for_object_in_video(
        #     print_results=False,
        # ))


asyncio.run(main())
