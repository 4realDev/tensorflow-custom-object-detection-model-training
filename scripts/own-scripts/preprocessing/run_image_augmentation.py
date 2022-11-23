
from typing import Literal, Optional, Union, List
from cv2 import Mat
import os
import tensorflow as tf
import cv2
import config

files = config.files
paths = config.paths


def augment_image_with_augmentation_operation(
        original_image: Mat,
        original_image_name: str,
        original_image_file_path: str,
        augmentation_operation: Literal['brightness', 'flip_left_right', 'saturation', 'contrast', 'crop_to_bounding_box'],
        augmentation_operation_values_array: Optional[List[Union[int, float]]] = None):

    # fmt: off
    # if not os.path.exists(os.path.join(paths['IMAGE_PATH'], f"{augmentation_operation}-augmented")):
    #     os.mkdir(os.path.join(paths['IMAGE_PATH'], f"{augmentation_operation}-augmented"))

    if os.path.isfile(original_image_file_path):
        augmented_image_data_array = []

        print(f"\n augmentation operation: {augmentation_operation}")

        if augmentation_operation_values_array == None:
            augmented_image = None
            if augmentation_operation == 'flip_left_right':
                augmented_image = tf.image.flip_left_right(original_image)
                augmented_image_name = f"{augmentation_operation}-augmented-{original_image_name}"
                augmented_image_data_array.append({"image": augmented_image, "name": augmented_image_name})
            elif augmentation_operation == 'crop_to_bounding_box':
                original_img_height = original_image.shape[0]
                original_img_width = original_image.shape[1]
                cropped_img_height = int(original_img_height/2)
                cropped_img_width = int(original_img_width/2)

                augmented_image_top_left_quatre = tf.image.crop_to_bounding_box(
                    image = original_image, 
                    offset_height = 0, 
                    offset_width = 0, 
                    target_height = cropped_img_height, 
                    target_width = cropped_img_width
                )

                augmented_image_top_right_quatre = tf.image.crop_to_bounding_box(
                    image = original_image, 
                    offset_height = 0, 
                    offset_width = cropped_img_width, 
                    target_height = cropped_img_height, 
                    target_width = cropped_img_width
                )

                augmented_image_bottom_left = tf.image.crop_to_bounding_box(
                    image = original_image, 
                    offset_height = cropped_img_height, 
                    offset_width = 0, 
                    target_height = cropped_img_height, 
                    target_width = cropped_img_width)

                augmented_image_bottom_right = tf.image.crop_to_bounding_box(
                    image = original_image, 
                    offset_height = cropped_img_height, 
                    offset_width = cropped_img_width, 
                    target_height = cropped_img_height, 
                    target_width = cropped_img_width)

                augmented_image_middle = tf.image.crop_to_bounding_box(
                    image = original_image, 
                    offset_height = cropped_img_height - int(cropped_img_height/2), 
                    offset_width = cropped_img_width - int(cropped_img_width/2), 
                    target_height = cropped_img_height, 
                    target_width = cropped_img_width)

                augmented_image_name = f"{augmentation_operation}-augmented-{original_image_name}"
                augmented_image_data_array.append({"image": augmented_image_top_left_quatre, "name": f"top_left-{augmented_image_name}"})
                augmented_image_data_array.append({"image": augmented_image_top_right_quatre, "name": f"top-right-{augmented_image_name}"})
                augmented_image_data_array.append({"image": augmented_image_bottom_left, "name": f"bottom-left-{augmented_image_name}"})
                augmented_image_data_array.append({"image": augmented_image_bottom_right, "name": f"bottom-right-{augmented_image_name}"})
                augmented_image_data_array.append({"image": augmented_image_bottom_right, "name": f"bottom-right-{augmented_image_name}"})
                augmented_image_data_array.append({"image": augmented_image_middle, "name": f"middle-{augmented_image_name}"})
            else:
                print(f"Error: The augmentation_operation {augmentation_operation} does not exist in augment_image_with_augmentation_operation().")
                return

        else:
            for augmentation_operation_value in augmentation_operation_values_array:
                augmented_image = None
                if augmentation_operation == 'brightness':
                    augmented_image = tf.image.adjust_brightness(original_image, delta=augmentation_operation_value)

                elif augmentation_operation == 'saturation':
                    augmented_image = tf.image.adjust_saturation(original_image, saturation_factor=augmentation_operation_value)

                elif augmentation_operation == 'contrast':
                    augmented_image = tf.image.adjust_contrast(original_image, contrast_factor=augmentation_operation_value)

                else:
                    print(f"Error: The augmentation_operation {augmentation_operation} does not exist in augment_image_with_augmentation_operation().")
                    return

                augmented_image_name = f"{str(augmentation_operation_value)}-{augmentation_operation}-augmented-{original_image_name}"
                augmented_image_data_array.append({"image": augmented_image, "name": augmented_image_name})


        if len(augmented_image_data_array) > 0:
            for augmented_image_data in augmented_image_data_array:
                augmented_image = augmented_image_data['image']
                augmented_image_name = augmented_image_data['name']

                # augmented_image_file_path = os.path.join(
                #     paths['IMAGE_PATH'], f"{augmentation_operation}-augmented", augmented_image_name)
                # result: bool = cv2.imwrite(
                # augmented_image_file_path, augmented_image.numpy())
                result: bool = cv2.imwrite(
                    os.path.join(paths['AUGMENTED_IMAGE_PATH'], augmented_image_name), augmented_image.numpy())

                if result == True:
                    print(f"File {augmented_image_name} saved successfully")
                else:
                    print(f"Error in saving file {augmented_image_name}")

    else:
        print(f"Error: Path to {original_image_file_path} is not a regular file.")
    # fmt: on


def img_augumentation(images_path):

    # get the path or directory
    augmented_images_array: List[Mat] = []
    for image_name in os.listdir(images_path):

        # check if image_name ends with png or jpg or jpeg
        if (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
            image_file_path = os.path.join(
                images_path, image_name)
            # returns numpy.ndarray
            image_array = cv2.imread(image_file_path)

            # Change brightness of image with tf.image.adjust_brightness by providing brightness factor
            augment_image_with_augmentation_operation(
                original_image=image_array,
                original_image_name=image_name,
                original_image_file_path=image_file_path,
                augmentation_operation='brightness',
                augmentation_operation_values_array=[-0.3, -0.15, 0.15, 0.3]
            )

            # Flip image horizontally (left to right) with tf.image.flip_left_right
            augment_image_with_augmentation_operation(
                original_image=image_array,
                original_image_name=image_name,
                original_image_file_path=image_file_path,
                augmentation_operation='flip_left_right',
            )

            # Saturate image with tf.image.adjust_saturation by providing saturation factor
            augment_image_with_augmentation_operation(
                original_image=image_array,
                original_image_name=image_name,
                original_image_file_path=image_file_path,
                augmentation_operation='saturation',
                augmentation_operation_values_array=[
                    0.5, 0.75, 1.25, 1.5, 1.75, 2]
            )

            augment_image_with_augmentation_operation(
                original_image=image_array,
                original_image_name=image_name,
                original_image_file_path=image_file_path,
                augmentation_operation='contrast',
                augmentation_operation_values_array=[
                    0.5, 0.75, 1.25, 1.5, 1.75, 2]
            )

            augment_image_with_augmentation_operation(
                original_image=image_array,
                original_image_name=image_name,
                original_image_file_path=image_file_path,
                augmentation_operation='crop_to_bounding_box',
            )


def main():
    img_augumentation(os.path.join(paths['IMAGE_PATH'], "augmentation_test"))


if __name__ == '__main__':
    main()
