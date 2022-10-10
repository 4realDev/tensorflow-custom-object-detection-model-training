# python scrape_google_images_chrome.py -t bearing -n 5 -dp C:\_WORK\GitHub\_data-science\ImageScraping\executables\chromedriver.exe -tp C:\_WORK\GitHub\_data-science\TFODCourse\Tensorflow\workspace\images\web_scraped_images

# MAIN SOURCE: https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d
# MODIFICATION:
# 1. adjustung selenium.find_element_by_xy method to newer methods
# 2. using selenium expected_conditions methods two be faster and more efficient (instead of time.sleep)
# 3. remove fixed time.sleep values and add logic to wait till full res image has loaded correctly
# 4. add logic to accept cookies at the beginning so script can run correctly
# 5. remove CSS_SELECTORS with better readable CLASS_NAMES
# 6. give file names counter inside their names
# 7. split logic in more functions for better readability

# DESCIPTION:
# images seen on the html google image search page are low resolution thumbnail images
# full resolution images can be obtained by clicking on the image
# all images are stored in identical containers of html code
# URL = link to image file
# Goal: get url for each image presentend on page

# necessary for pillow image operations
#!pip install Pillow
import re
from PIL import Image

import io
import os
import argparse

# allows to make HTTP get request to an URL to get the src of image
# grab data of the image we want to download
#!pip install requests
import requests
import time

# Selenium: used to automate web browser interactions with Python (pretends to be real user, opens browser, "moves" cursor, clicks buttons / images)
# allows to search for specific tags, classNames, texts .. within a HTML-DOM
# Initial idea behind selenium: automated testing, but equally powerful when it comes to automating repetitive web-based task
#!pip install selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

parser = argparse.ArgumentParser(
    description="Scrap full resolution image urls of google image search in chrome and download them into images folder.")
parser.add_argument('-t', '--search_term', metavar='search_term', type=str,
                    help='enter the search term for the google image search', required=True)
parser.add_argument('-n', '--fetched_images_number', metavar='fetched_images_number', type=int,
                    help='enter the number of images you want to fetch', required=True)
parser.add_argument('-dp', '--driver_path', metavar='driver_path', type=str,
                    help='enter the path to your chrome web driver (you need to download the chrome web driver corresponding to your chrome web browser version)', required=True)
parser.add_argument('-tp', '--target_path', metavar='target_path', type=str,
                    help='enter the path where the scraped images should be downloaded')
args = parser.parse_args()

# driver_path = "C:\_WORK\GitHub\_data-science\ImageScraping\executables\chromedriver.exe"


def scroll_to_end(wd: webdriver, sleep_between_interactions: int = 1):
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # use small sleep_between_interactions to load remainder of the images
    time.sleep(sleep_between_interactions)


def click_accept_cookies_button(wd: webdriver):
    # accept cookies to load page
    accept_cookies_button = WebDriverWait(wd, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "Nc7WLe")))
    accept_cookies_button.click()


def click_show_more_results_button_if_it_exists(wd: webdriver):
    load_more_button = WebDriverWait(wd, 10).until(
        EC.presence_of_element_located(
            (By.CLASS_NAME, "mye4qd"))
    )
    # use js execute_script, because input field cannot be clicked via clicked() method from selenium
    if load_more_button:
        wd.execute_script(
            "document.querySelector('.mye4qd').click();")


# https://www.youtube.com/watch?v=Yt6Gay8nuy0
def wait_for_full_res_image_and_extract_its_url(thumbnail, wd: webdriver):
    # seeing blurry image before correct image loads
    # disconnecting from internet shows, that google does not load the hi-res image, but instead stays on the already loaded thumbnail
    # conclusion: images that are immediatly displayed after clicking on thumbnail, are the same as the low-res thumbnail
    # goal: found out, how long to wait, until the hi-res image is loaded
    # continually grabbing src of loaded preview image and bigger image and checking if they are the same
    # if they are not the same (full-res image has loaded), grab src from bigger image and continue clicking
    # constant waiting time would not be efficient, since full-res image loading depends on multiple factors (image size, internet connection speed ...)

    # it does not always load full-res image for super-high-res images or loading failes while execution
    # therefore time parameter was added for timeout when it cannot be loaded for more then 10 secs
    # starting while True loop to wait until the URL inside large image view is different from preview one

    thumbnail_url = thumbnail.get_attribute('src')
    timeStarted = time.time()

    while True:
        # get clicked, scaled images by their common class name "n3VNCb"
        # could potentially return multiple image -> look through all and find valid src
        full_size_images = WebDriverWait(wd, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "n3VNCb")))

        # print("Waiting for full res image...")

        for full_size_image in full_size_images:
            full_size_image_url = full_size_image.get_attribute('src')

            # high res image has loaded
            # check if image has src with http inside the image src (valid link to download the image)
            # check if image src is new one (different from low res thumbnail image src)
            if full_size_image_url and 'http' in full_size_image_url and full_size_image_url != thumbnail_url:
                print("Successfully loaded full res image")
                return full_size_image_url
            # high res image has not loaded yet, image url is still the one of the thumbnail
            else:
                # making timeout if full res image cannot be loaded
                currentTime = time.time()
                if currentTime - timeStarted > 10:
                    print("Timeout on " + thumbnail_url +
                          "! Program downloads lower resolution image and moves on to next image.")
                    # return low res url of thumbnail image
                    return thumbnail_url


def fetch_image_urls(search_term: str, max_links_to_fetch: int, wd: webdriver):
    # build the google search_term and load the page
    # TODO: analyse this search_term and understand why this is used instead of:
    # url = "https://www.google.com/search?q=" + search_search_term + \
    # "&sxsrf=ALiCzsZoI_HzLXbJ3QgCAcmTXsfMPMypCA:1658360543832&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjpof-40oj5AhVPSkEAHSJAASwQ_AUoAXoECAQQAw&biw=960&bih=879&dpr=1"
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
    # load page with chrome web driver
    wd.get(search_url.format(q=search_term))
    wd.maximize_window()  # For maximizing window

    click_accept_cookies_button(wd)

    image_urls = set()  # use set of urls to ensure no duplicate urls
    image_count = 0  # total number of loaded images / fetched image urls
    # dynamic starting point of the thumbnail addition
    # gets updated with each new while loop to the new number of already added thumbnails (after scrolling down and loading more thumbnails)
    results_start = 0

    while image_count < max_links_to_fetch:
        # scroll to end of page to load more images as long as condition image_count < max_links_to_fetch is true
        scroll_to_end(wd)

        # find all thumbnails with class name "Q4LuWd"
        thumbnail_results = WebDriverWait(wd, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "Q4LuWd")))

        number_results = len(thumbnail_results)

        print(
            f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for thumbnail in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                thumbnail.click()
            # could get error while clicking on image, but don't interrupt whole script even if error occures
            except Exception:
                # go to next item in thumbnails
                continue

            # method works similar to time.sleep, because of while loop
            new_image_url = wait_for_full_res_image_and_extract_its_url(
                thumbnail, wd)
            image_urls.add(new_image_url)
            image_count = len(image_urls)

            if image_count >= max_links_to_fetch:
                print(f"Found: {image_count} image links, done!")
                break
            else:
                print("Found:", image_count,
                      "image links, looking for more ...")
                click_show_more_results_button_if_it_exists(wd)

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls

# TODO: Evaluate this code snipped
# This can be done simply without PIL nor io, just as “open(file_name, “wb”).write(requests.get(url).content)”
# Download and save image


def download_image(target_path: str, url: str, file_name: str):
    if (os.path.exists(os.path.join(target_path,  file_name))):
        print(f"WARNING - File {file_name} already exists and will be skipped")
    else:
        try:
            # get content of image with HTTP get request to URL
            image_content = requests.get(url).content

        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")

        try:
            # convert image_content in byte IO data type (binary data)
            # to store it as binary data type directly in computers memory
            image_file = io.BytesIO(image_content)

            # convert binary data to pill image, so it can be saved
            # allows to easly save image using image.save
            # TODO: understand convert('RGB')
            image = Image.open(image_file).convert('RGB')

            # generate file path
            # file_path = os.path.join(target_path, hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
            file_path = os.path.join(target_path, file_name)

            # open new file at file_path in wb node (write bytes)
            # so we can write bytes to this file
            # wb = write bytes
            with open(file_path, 'wb') as f:
                # save / write the bytes of wanted image to new open file inside file_path f as JPEG
                # TODO: check if quality=85 is useful
                image.save(f, "JPEG", quality=85)
            print(f"SUCCESS - saved {url} - as {file_path}")

        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")


def search_and_download(search_term: str, driver_path: str, number_images: int, target_path='./images'):
    try:
        os.chdir(target_path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(target_path))
    except NotADirectoryError:
        print("{0} is not a directory".format(target_path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(
            target_path))

    # '_'.join(...) to convert result of .split to str
    # TypeError: join() argument must be str, bytes, or os.PathLike object, not 'list'

    # get webdriver
    # opens google chrome window / webdriver has started
    # "Chrome is being controlled by automated test driver."
    # write controls of browser into driver variable for interacting with automated browser
    with webdriver.Chrome(executable_path=driver_path) as wd:
        fetched_img_urls = fetch_image_urls(
            search_term, number_images, wd=wd)

    for fetched_img_url in fetched_img_urls:
        download_image(target_path, fetched_img_url,
                       #    search_term + "_" + str(i+1) + ".jpg"
                       re.sub('[^A-Za-z0-9 ]+', '', fetched_img_url) + ".jpg")

    print("FINISH IMAGE DOWNLOADING PROCESS")


# Paste executable as PATH into virtual environment to reference it from python script
# web driver executable files: allows to autamate appropriate web browser (chrome)
search_and_download(search_term=args.search_term,
                    driver_path=args.driver_path, number_images=args.fetched_images_number, target_path=args.target_path)
