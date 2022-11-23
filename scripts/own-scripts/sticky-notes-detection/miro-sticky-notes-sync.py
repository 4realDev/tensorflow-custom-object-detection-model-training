# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro-sticky-notes-sync.py

# fmt: off
import os
from typing import Tuple
import cv2
from genericpath import exists
import aiohttp
import asyncio
import nest_asyncio

import sys

import requests
sys.path.insert(
    1, '..\scripts\own-scripts\preprocessing')

import config

from miro_rest_api_functions import \
    get_all_items, \
    create_frame, \
    create_sticky_note, \
    create_line, \
    create_image, \
    create_new_miro_board_or_get_existing

from miro_tfod_functions import \
    get_image_with_overlayed_labeled_bounding_boxes, \
    get_detections_from_img, \
    get_bounding_boxes_above_min_score_thresh, \
    load_latest_checkpoint_of_custom_object_detection_model, \
    scan_for_object_in_video

from miro_ocr_functions import \
    get_image_ocr_data, \
    extract_string_out_of_ocr_data, \
    rename_cropped_image_with_ocr_string, \
    get_image_with_overlayed_ocr_bounding_boxes_and_text, \
    save_image_in_folder, \
    save_image_with_detection_for_debug, \
    get_timestamp_yyyy_mm_dd_hh_mm_ss, \
    create_timestamp_folder_and_return_its_path, \
    crop_image_to_bounding_boxes

import argparse

parser = argparse.ArgumentParser(
    description="Digitilize sticky notes inside Miro.")
parser.add_argument('-n', '--miro_board_name', metavar='miro_board_name', type=str,
                    help='Enter the miro board name in which the digitalized sticky note snapshot should be saved. Entering an already existing name, will result in a new frame with the snapshot within the same miro board. Entering a yet non-existing name, will result in the creation of a new miro board with this name.', required=False)
parser.add_argument('-s', '--save_in_existing_miro_board', metavar='save_in_existing_miro_board', type=bool,
                    help='Enter the flag, if the miro board, given by its miro board name, already exists and the snapshot should be saved in the existing miro board, or if a new miro board should be created for the snapshot frame.', required=False, default=False)
args = parser.parse_args()

nest_asyncio.apply()
# fmt: on

# TODO: Replace cv2.show logic with debug/controlling folder instead
# Save all images with OCR recognition and TFOD object recognition
# Save autolabelling output images


files = config.files
paths = config.paths
ocr_confidence_threshold = config.ocr_confidence_threshold
save_image_overlayed_with_ocr_visualization = config.save_image_overlayed_with_ocr_visualization
save_original_image_overlayed_with_labeled_detection_bounding_boxes = config.save_original_image_overlayed_with_labeled_detection_bounding_boxes

min_score_thresh = config.min_score_thresh


async def get_recognized_sticky_notes_data(img_file_path, timestamped_folder_name: str) -> Tuple[list, str]:
    timestamped_folder_path = create_timestamp_folder_and_return_its_path(
        timestamped_folder_name)

    img = cv2.imread(img_file_path)

    img_detections = get_detections_from_img(img)

    img_detection_data_list = get_bounding_boxes_above_min_score_thresh(
        detections=img_detections,
        imgHeight=img.shape[0],
        imgWidth=img.shape[1],
        min_score_thresh=min_score_thresh
    )

    # necessary to save the cropped image as files, to run PaddleOCR detection on them
    # "position": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
    # "color": color,
    # "name": cropped_img_according_to_its_bounding_box_name,
    # "ocr_recognized_text": "",
    cropped_bounding_boxes_images_data = crop_image_to_bounding_boxes(
        img,
        img_detection_data_list,
    )
    print("\n")

    # Save cropped images of the detected objects inside the bounding boxes in dedicated backup folder
    # necessary for OCR, because OCR does only work with path of existing image file
    for cropped_image in cropped_bounding_boxes_images_data:
        save_image_in_folder(
            cropped_image['image'],
            cropped_image['name'],
            timestamped_folder_path
        )

    print("\n")

# 1. Get all images in the timestamped_folder_path
# 2. For each cropped image of the sticky notes inside the timestamped_folder_path: Gets the ocr data array
# 3. Loop through the image_ocr_data_array, take every ocr bounding box and
#    create one single string out of all ocr bounding boxes texts for each cropped image of the sticky notes
# 4. Change the name of the cropped image to the new created single string of the image's ocr text
# 5. Get the dominant color of every cropped image and its detected sticky note
#    and assign it to a color class for sticky notes, which miro can understand (with the help of HSV color model)
# 6. Override cropped_image_data with new data from text and color recognition

    # For each cropped image of the sticky notes inside the timestamped_folder_path
    for index, cropped_image in enumerate(cropped_bounding_boxes_images_data):
        # Get (saved) cropped image of sticky note and check if it is a existing file
        cropped_image_file_path = os.path.join(
            timestamped_folder_path, cropped_image['name'])
        if os.path.isfile(cropped_image_file_path):
            # Get the ocr data array for this image
            # [{
            #   "position": {"xmin": int, "ymin": int, "xmax": int, "ymax": int},
            #   "text": str
            # }]
            # (use euclidean distance to sort the words in the correct order from top left to bottom right)
            ocr_recognized_text_and_boxes_array = await asyncio.create_task(
                get_image_ocr_data(
                    cropped_image_file_path,
                    ocr_confidence_threshold,
                )
            )
            cropped_image['ocr_data'] = ocr_recognized_text_and_boxes_array
            # Loop through ocr_recognized_text_and_boxes_array, take every ocr bounding box and create single string
            # out of all ocr bounding boxes texts for each cropped image of the sticky notes
            # ()
            cropped_image['ocr_recognized_text'] = extract_string_out_of_ocr_data(
                cropped_image)

            # Rename the cropped image file name to the image's recognized ocr text
            # (remove forbidden printable ASCII characters and add timestamp in front of name to prevent same names)
            [cropped_image_new_ocr_name, cropped_image_new_ocr_path] = rename_cropped_image_with_ocr_string(
                cropped_image,
                index,
                cropped_image_file_path,
                timestamped_folder_path
            )

            # Override cropped_image name with new set file name
            # and add path attribute to post the cropped image of the sticky note later inside the miro board
            cropped_image['name'] = cropped_image_new_ocr_name
            cropped_image['path'] = cropped_image_new_ocr_path

            if save_image_overlayed_with_ocr_visualization:
                img_with_ocr_visualization = get_image_with_overlayed_ocr_bounding_boxes_and_text(
                    cropped_image['path'], cropped_image['ocr_data'])

                save_image_in_folder(
                    img_with_ocr_visualization, f"ocr-visualization-{cropped_image['name']}.png", timestamped_folder_path)

    print("\n")

    # Save the original image in dedicated backup folder (for reference)
    # necessary to save the original image as a file, to post it inside miro
    save_image_in_folder(
        img,
        f"_{timestamped_folder_name}-original.png",
        timestamped_folder_path,
    )

    # Save the original image with the detection bounding boxes of the sticky notes
    # inside the dedicated backup folder (for manual detection control)
    if save_original_image_overlayed_with_labeled_detection_bounding_boxes:
        # only necessary for saving the image with the detections inside the backup folder
        img_with_overlayed_labeled_detection_bounding_boxes = get_image_with_overlayed_labeled_bounding_boxes(
            img,
            img_detections,
        )

        save_image_in_folder(
            img_with_overlayed_labeled_detection_bounding_boxes,
            f"_{timestamped_folder_name}-with_bounding_boxes.png",
            timestamped_folder_path,
        )

    return cropped_bounding_boxes_images_data


async def run_miro_sync_process(img_file_path: str, session):
    miro_board_name = ""
    timestamped_folder_name = ""
    timestamped_frame_name = ""
    timestamp = get_timestamp_yyyy_mm_dd_hh_mm_ss()

    if args.miro_board_name == None:
        miro_board_name = timestamp
        timestamped_folder_name = timestamp
    else:
        miro_board_name = args.miro_board_name
        timestamped_folder_name = f"{miro_board_name}-{timestamp}"

    timestamped_frame_name = timestamped_folder_name

    # TODO: Remove/Replace Code when start testing with Video Frames
    try:
        with open(img_file_path) as f:
            print("\nFile present\n")
    except FileNotFoundError:
        print('\nFile is not present\n')

    # HERE
    # Create miro board or get id of exsting one, if timestamp name already exists
    board_id = await asyncio.create_task(
        create_new_miro_board_or_get_existing(
            name=miro_board_name,
            description=miro_board_name,
            save_in_existing_miro_board=args.save_in_existing_miro_board,
            session=session
        )
    )

    # Savety check for MIRO REST API behaviour for getting all boards, while new created board is maybe still in query for indexing
    # "If you use any other filter (then teamId), you need to give a few seconds for the indexing of newly created boards before retrieving boards."
    if board_id == "-1":
        return
    print("\n")
    # HERE

    # Use TFOD, PaddleOCR, and cv2 logic to get the recognized sticky notes data in the given image from the passed img_file_path with the following information:
    # bounding-box position         | sticky_note_data['position'] -> {"xmin": int, "xmax": int, "ymin": int, "ymax": int}
    # recognized color              | sticky_note_data['color'] -> str
    # detected ocr-text             | sticky_note_data['ocr_recognized_text'] -> str
    # path where image was saved    | sticky_note_data['path'] -> str
    # file name of saved image      | sticky_note_data['name'] -> str
    sticky_notes_data = await asyncio.create_task(get_recognized_sticky_notes_data(img_file_path, timestamped_folder_name))
    # return

    if len(sticky_notes_data) == 0:
        print("ERROR: No sticky notes could be detected.")
        return

    # Calculate average sticky note size for miro board
    average_sticky_note_width = 0
    for sticky_note_data in sticky_notes_data:
        sticky_note_width = sticky_note_data['position']['xmax'] - \
            sticky_note_data['position']['xmin']
        average_sticky_note_width += sticky_note_width

    average_sticky_note_width = average_sticky_note_width / \
        len(sticky_notes_data)

    full_comparison_img = cv2.imread(img_file_path)
    full_comparison_img_height = full_comparison_img.shape[0]
    full_comparison_img_width = full_comparison_img.shape[1]

    padding_between_stickies_and_comparison_image: int = 0
    frame_width = full_comparison_img_width * 3 + \
        padding_between_stickies_and_comparison_image * 2

    padding_between_frames = 1000
    allFrameWidgets = await asyncio.create_task(get_all_items(
        item_type="frame",
        max_num_of_items=10,
        board_id=board_id,
        session=session
    ))

    allFrameWidgetsLength = 0
    for frameWidget in allFrameWidgets:
        allFrameWidgetsLength = allFrameWidgetsLength + \
            frameWidget['geometry']['width']

    frameOffset = len(allFrameWidgets) * \
        padding_between_frames + allFrameWidgetsLength

    # Create frame for this timestamp for store all sticky notes inside
    frame_id = await asyncio.create_task(create_frame(
        pos_x=frameOffset,
        pos_y=0,
        title=timestamped_frame_name,
        height=full_comparison_img_height,
        width=frame_width,
        board_id=board_id,
        session=session
    ))

    create_sticky_notes_status_code_array = []
    for sticky_note_data in sticky_notes_data:
        create_sticky_notes_status_code = await asyncio.create_task(create_sticky_note(
            pos_x=sticky_note_data['position']['xmin'] +
            average_sticky_note_width / 2,
            pos_y=sticky_note_data['position']['ymin'] +
            average_sticky_note_width / 2,
            width=average_sticky_note_width,
            color=sticky_note_data['color'],
            text=sticky_note_data['ocr_recognized_text'],
            board_id=board_id,
            parent_id=frame_id,
            session=session)
        )
        create_sticky_notes_status_code_array.append(
            create_sticky_notes_status_code)
    if all(create_sticky_notes_status_code == requests.codes.created for create_sticky_notes_status_code in create_sticky_notes_status_code_array):
        print(
            f"Successfully created {len(sticky_notes_data)} sticky notes inside the miro board {miro_board_name}.")

    await asyncio.create_task(
        create_line(
            pos_x=full_comparison_img_width + padding_between_stickies_and_comparison_image / 2,
            pos_y=full_comparison_img_height / 2,
            width=8,
            height=full_comparison_img_height,
            color='#000000',
            board_id=board_id,
            parent_id=frame_id,
            session=session
        )
    )

    create_sticky_notes_images_status_code_array = []
    additional_second_column_x_distance = full_comparison_img_width + \
        padding_between_stickies_and_comparison_image
    for sticky_note_data in sticky_notes_data:
        create_sticky_notes_images_status_code = await asyncio.create_task(create_image(
            pos_x=sticky_note_data['position']['xmin'] +
            average_sticky_note_width / 2 + additional_second_column_x_distance,
            pos_y=sticky_note_data['position']['ymin'] +
            average_sticky_note_width / 2,
            width=average_sticky_note_width,
            title=sticky_note_data['name'],
            path=sticky_note_data['path'],
            board_id=board_id,
            parent_id=frame_id,
            session=session)
        )
        create_sticky_notes_images_status_code_array.append(
            create_sticky_notes_images_status_code)
    if all(create_sticky_notes_status_code == requests.codes.created for create_sticky_notes_status_code in create_sticky_notes_status_code_array):
        print(
            f"Successfully created {len(sticky_notes_data)} images of the {len(sticky_notes_data)} sticky notes inside the miro board {miro_board_name}.")

    await asyncio.create_task(
        create_line(
            pos_x=(full_comparison_img_width +
                   padding_between_stickies_and_comparison_image / 2) * 2,
            pos_y=full_comparison_img_height / 2,
            width=8,
            height=full_comparison_img_height,
            color='#000000',
            board_id=board_id,
            parent_id=frame_id,
            session=session
        )
    )

    additional_third_column_x_distance = full_comparison_img_width * 2 + \
        padding_between_stickies_and_comparison_image * 2 - full_comparison_img_width / 2

    # +7px to ensure that the line is not laying below or above the image
    create_full_comparison_image_status_code = await asyncio.create_task(create_image(
        pos_x=full_comparison_img_width + additional_third_column_x_distance + 7,
        pos_y=full_comparison_img_height/2,
        width=full_comparison_img_width,
        title=f"{miro_board_name}-full-image",
        path=img_file_path,
        board_id=board_id,
        parent_id=frame_id,
        session=session)
    )
    if create_full_comparison_image_status_code == requests.codes.created:
        print(
            f"Successfully created the original images of the {len(sticky_notes_data)} sticky notes inside the miro board {miro_board_name}.")
    # fmt: on


async def main():
    async with aiohttp.ClientSession() as session:
        load_latest_checkpoint_of_custom_object_detection_model()

        # img_file_path_1 = r"C:\Users\vbraz\Desktop\RADAR_SYNC_IMAGES\IMG_5553.jpg"
        # img_file_path_2 = r"C:\Users\vbraz\Desktop\RADAR_SYNC_IMAGES\IMG_5557.jpg"
        # img_file_path_3 = r"C:\Users\vbraz\Desktop\RADAR_SYNC_IMAGES\IMG_5564.jpg"
        # img_file_path_4 = r"C:\Users\vbraz\Desktop\RADAR_SYNC_IMAGES\IMG_5566.jpg"
        # img_file_path_5 = r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\randy-bachelor-sticky-notes-images\IMG_0249.jpg"
        img_file_path_6 = r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\randy-bachelor-sticky-notes-images\IMG_0265.jpg"
        # img_file_path_7 = r"C:\_WORK\GitHub\_data-science\TensorFlow\workspace\my_sticky_notes_ssd_resnet50_v1_fpn_640x640_coco17_tpu-8_25000\images\crop_to_bounding_box-augmented\middle-crop_to_bounding_box-augmented-image.jpeg"
        await asyncio.create_task(run_miro_sync_process(img_file_path_6, session))
        # await asyncio.create_task(run_miro_sync_process(img_file_path_2, session))
        # await asyncio.create_task(run_miro_sync_process(img_file_path_3, session))
        # await asyncio.create_task(run_miro_sync_process(img_file_path_4, session))
        # await asyncio.create_task(run_miro_sync_process(img_file_path_5, session))
        # await asyncio.create_task(run_miro_sync_process(img_file_path_6, session))

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
