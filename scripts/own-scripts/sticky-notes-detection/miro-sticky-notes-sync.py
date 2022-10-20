# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro-sticky-notes-sync.py

# fmt: off
import tensorflow as tf
import cv2
import numpy as np
from genericpath import exists
from datetime import date, datetime
import os
import aiohttp
import asyncio
import nest_asyncio

import sys
sys.path.insert(
    1, 'C:\_WORK\GitHub\_data-science\TensorFlow\scripts\own-scripts\preprocessing')
import config

from miro_rest_api_functions import \
    get_all_miro_board_names_and_ids, \
    get_all_items, \
    delete_item, \
    delete_all_items, \
    create_frame, \
    create_item, \
    create_line, \
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

nest_asyncio.apply()
# fmt: on


files = config.files
paths = config.paths

global_session = None
global_board_id = None


async def get_recognized_sticky_notes_data(img_file_path):
    timestamp = get_timestamp_yyyy_mm_dd_hh_mm_ss()
    timestamped_folder_path = create_timestamp_folder_and_return_its_path(
        timestamp)
    img = cv2.imread(img_file_path)

    img_detections = get_detections_from_img(img)

    img_detection_bounding_boxes = get_bounding_boxes_above_min_score_thres(
        detections=img_detections, imgHeight=img.shape[0], imgWidth=img.shape[1])

    img_with_detection_bounding_boxes = visualize_detections_from_image(
        img, img_detections, visualize_bounding_box_detections_in_image=False)

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
        load_latest_checkpoint_of_custom_object_detection_model()

        # TODO: Remove/Replace Code when start testing with Video Frames
        img_file_path = r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG"
        try:
            with open(img_file_path) as f:
                print("File present")
        except FileNotFoundError:
            print('\n\n\nFile is not present\n\n\n')

        # Use TFOD, PaddleOCR, and cv2 logic to get the recognized sticky notes in the image with their:
        # bounding-box position         | sticky_note_data['position'] -> {"xmin": int, "xmax": int, "ymin": int, "ymax": int}
        # recognized color              | sticky_note_data['color'] -> str
        # detected ocr-text             | sticky_note_data['ocr_recognized_text'] -> str
        # path where image was saved    | sticky_note_data['path'] -> str
        # file name of saved image      | sticky_note_data['name'] -> str
        [sticky_notes_data, timestamp] = await asyncio.create_task(get_recognized_sticky_notes_data(img_file_path))

        # Create miro board or get id of exsting one, if timestamp name already exists
        board_id = await asyncio.create_task(create_new_miro_board_or_get_existing(name=timestamp, description=timestamp, session=session))
        print(f"board_id: {board_id}")

        # Calculate average sticky note size for miro board
        average_sticky_note_width = 0
        for sticky_note_data in sticky_notes_data:
            sticky_note_width = sticky_note_data['position']['xmax'] - \
                sticky_note_data['position']['xmin']
            average_sticky_note_width += sticky_note_width

        average_sticky_note_width = average_sticky_note_width / \
            len(sticky_notes_data)

        # Get top left and bottom right position of all sticky notes objects
        # in sticky_notes_data the bounding boxes differ in size, therefore their ymax and xmax differ from what we would expect in miro
        # to normalize the size of the xmax and ymax, we use the maximum xmin value and add the average_sticky_note_width
        # fmt: off
        # min_xmin_of_sticky_notes_data = min([sticky_note_data['position']['xmin'] for sticky_note_data in sticky_notes_data])
        # min_ymin_of_sticky_notes_data = min([sticky_note_data['position']['ymin'] for sticky_note_data in sticky_notes_data])
        # max_xmax_of_sticky_notes_data = max([sticky_note_data['position']['xmin'] for sticky_note_data in sticky_notes_data]) + average_sticky_note_width
        # max_ymax_of_sticky_notes_data = max([sticky_note_data['position']['ymin'] for sticky_note_data in sticky_notes_data]) + average_sticky_note_width
        # all_sticky_notes_width = max_xmax_of_sticky_notes_data - min_xmin_of_sticky_notes_data

        full_comparison_img = cv2.imread(img_file_path)
        full_comparison_img_height = full_comparison_img.shape[0]
        full_comparison_img_width = full_comparison_img.shape[1]

        padding_between_stickies_and_comparison_image: int = 0
        frame_width = full_comparison_img_width * 3 + padding_between_stickies_and_comparison_image * 2

        # Create frame for this timestamp for store all sticky notes inside

        # comparison_img_aspect_ratio: float = img.shape[1] / img.shape[0]
        # comparison_image_width: int = frame_height
        # comparison_img_height: int = comparison_image_width / comparison_img_aspect_ratio
        # print(f"aspect ratio: {comparison_img_aspect_ratio}")
        # print(f"width: {comparison_image_width}")   # 2348.4166666666674
        # print(f"height: {comparison_img_height}")   # 1761.3125

        frame_id = await asyncio.create_task(create_frame(
            pos_x = 0,
            pos_y = 0,
            title = str(timestamp),
            height = full_comparison_img_height,
            width = frame_width,
            board_id = board_id,
            session = session
        ))

        for sticky_note_data in sticky_notes_data:
            await asyncio.create_task(create_item(
                pos_x = sticky_note_data['position']['xmin'] + average_sticky_note_width / 2, 
                pos_y = sticky_note_data['position']['ymin'] + average_sticky_note_width / 2,
                width = average_sticky_note_width,
                color = sticky_note_data['color'],
                text = sticky_note_data['ocr_recognized_text'],
                board_id = board_id,
                parent_id = frame_id,
                session = session)
            )

        create_line(
            pos_x = full_comparison_img_width + padding_between_stickies_and_comparison_image / 2, 
            pos_y = full_comparison_img_height / 2,
            width = 8,
            height = full_comparison_img_height,
            color = '#000000',
            board_id = board_id,
            parent_id = frame_id,
            session = session
        )

        additional_second_column_x_distance = full_comparison_img_width + padding_between_stickies_and_comparison_image
        for sticky_note_data in sticky_notes_data:
            await asyncio.create_task(create_image(
                pos_x = sticky_note_data['position']['xmin'] + average_sticky_note_width / 2 + additional_second_column_x_distance,
                pos_y = sticky_note_data['position']['ymin'] + average_sticky_note_width / 2,
                width = average_sticky_note_width,
                title = sticky_note_data['name'],
                path = sticky_note_data['path'],
                board_id = board_id,
                parent_id = frame_id,
                session = session)
            )

        create_line(
            pos_x = (full_comparison_img_width + padding_between_stickies_and_comparison_image / 2) * 2, 
            pos_y = full_comparison_img_height / 2,
            width = 8,
            height = full_comparison_img_height,
            color = '#000000',
            board_id = board_id,
            parent_id = frame_id,
            session = session
        )

        additional_third_column_x_distance = full_comparison_img_width * 2  + padding_between_stickies_and_comparison_image * 2 - full_comparison_img_width / 2
        await asyncio.create_task(create_image(
            pos_x=full_comparison_img_width + additional_third_column_x_distance,
            pos_y=full_comparison_img_height/2,
            width=full_comparison_img_width,
            title=f"{timestamp}-full-image",
            path=img_file_path,
            board_id=board_id,
            parent_id=frame_id,
            session=session)
        )
        # fmt: on

        # min_xmin_of_sticky_notes_data = min(sticky_notes_data['position']['xmin'])
        # min_ymin_of_sticky_notes_data = min(sticky_notes_data['position']['ymin'])
        # max_xmax_of_sticky_notes_data = min(sticky_notes_data['position']['xmax'])
        # max_ymax_of_sticky_notes_data = min(sticky_notes_data['position']['ymax'])
        # frame_height = max_ymax_of_sticky_notes_data - min_ymin_of_sticky_notes_data
        # frame_width = max_xmax_of_sticky_notes_data - min_xmin_of_sticky_notes_data
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
