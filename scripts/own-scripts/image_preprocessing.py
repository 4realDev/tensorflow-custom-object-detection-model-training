# pip install requests
# pip install aiohttp
# pip install nest-asyncio
# cd Tensorflow\scripts & python miro-sticky-notes-sync.py

import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import asyncio
import aiohttp
import nest_asyncio
nest_asyncio.apply()


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

    # TODO: Should array not be outside of for loop
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


async def opencv_script_thresholding(img_path):
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

    await asyncio.create_task(get_image_text_data_by_ocr("C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).png", ocr_confidence_threshold, True))

    for i in range(5):
        plt.subplot(5, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

        plt.subplot(5, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

        plt.subplot(5, 3, i*3+3), plt.imshow((images[i*3+2]), 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()


global_session = None
global_board_id = None


async def main():
    async with aiohttp.ClientSession() as session:
        global global_session
        global_session = session
        opencv_script_thresholding(
            "C:\\Users\\vbraz\\Desktop\\sticky-notes-downloaded-images\\cando_sticky_notes\\image (1).jpeg")

asyncio.run(main())
