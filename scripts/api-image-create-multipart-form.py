import requests
import json

url = "https://api.miro.com/v2/boards/uXjVO7Ddvh0%3D/images"

headers = {
    "Accept": "application/json",
    # "Content-Type": "application/json",
    "Authorization": f"Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_N9OybOclP4WmwOKCNUjVuVMDshE"
}

# https: // developers.miro.com/docs/image_image-1
payload = {
    "data": {
        "title": "Test",
        "position": {
            "origin": "center",
            "x": 50,
            "y": 50,
        },
        "geometry": {
            "height": 230,
            "width": 230,
            "rotation": 0
        }
    }
}

payload2 = {
    "title": "Test",
    "position": {
        "origin": "center",
        "x": 50,
        "y": 50,
    },
    "geometry": {
        "height": 230,
        "rotation": 0
    }
}


img_path = r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG"

multipart_form_data = {
    'resource': ("test.png", open(r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb")),
    'data': (None, json.dumps(payload2), "application/json"),
}

response = requests.post(url, files=multipart_form_data, headers=headers)


# TEST 0
# https://stackoverflow.com/questions/12385179/how-to-send-a-multipart-form-data-with-requests-in-python
# multipart_form_data = {
#     'upload': ('custom_file_name.zip', open('myfile.zip', 'rb')),
#     'action': (None, 'store'),
#     'path': (None, '/path1')
# }
# response = requests.post('https://httpbin.org/post', files=multipart_form_data)

# multipart_form_data = {
#     'resource': ("test.png", open(r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb")),
#     'data': (None, '{"title for the image": "Test","position":{"x":100,"y":200,"origin":"center"},"geometry":{"width":100,"height":100,"rotation":0}"}'),
# }
# result: no error - but data is not recognized in miro

# multipart_form_data = {
#     'resource': ("test.png", open(r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb")),
#     'data': (None, payload),
# }
# result: TypeError: a bytes-like object is required, not 'dict'

# multipart_form_data = {
#     'resource': ('test.png', open(r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb")),
#     'data': (None, '"data": {"title": "title for the image","position": {"x": 100,"y": 200,"origin": "center"},"geometry": {"width": 100,"height": 100,"rotation": 0}}}'),
# }
# result: no error - but data is not recognized in miro

# multipart_form_data = {
#     'resource': {'test.png', open(r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb")},
#     'data': '{"title": "title for the image","position": {"x": 100,"y": 200,"origin": "center"},"geometry": {"width": 100,"height": 100,"rotation": 0}}'
# }
# result: error -> TypeError: a bytes-like object is required, not 'set'


# TEST 1 - WITH DATA
# up = {'image':(filename, open(filename, 'rb'), "multipart/form-data")}
# data = {"name": "foo", "point": 0.13, "is_accepted": False}
# request = requests.post(site, files=up, data=data)

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb"))},
#     data=json.dumps(({'data': {"title for the image": "Test", "position": {"x": 100, "y": 200,
#                                                                            "origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}}})),
#     headers=headers
# )
# result: error -> ValueError: Data must not be a string.

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb"))},
#     data={'data': (None, {"title for the image": "Test", "position": {"x": 100, "y": 200,
#                                                                       "origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}})},
#     headers=headers
# )

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb"))},
#     data={'data': (None, '"title for the image": "Test", "position": {"x": 100, "y": 200,"origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}')},
#     headers=headers
# )

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb"))},
#     data={'data': '"title for the image": "Test", "position": {"x": 100, "y": 200,"origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}'},
#     headers=headers
# )

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb"))},
#     data={'data': '{"title for the image": "Test", "position": {"x": 100, "y": 200,"origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}}'},
#     headers=headers
# )

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb"))},
#     data={'data': '{"data": {"title for the image": "Test", "position": {"x": 100, "y": 200,"origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}}}'},
#     headers=headers
# )

# response = requests.post(
#     url=url,
#     files={'resource': ("test.png", open(img_path, "rb")),
#            'data': '{"data": {"title for the image": "Test", "position": {"x": 100, "y": 200,"origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}}}'},
#     headers=headers
# )
# result: no error - but data is not recognized in miro


# response = requests.post(
#     url=url,
#     files={'resource': ("IMG_8693.jpeg", open(img_path, "rb"))},
#     data={'data': ("json.txt", open(
#         r"C:\_WORK\GitHub\_data-science\TensorFlow\scripts\json.txt", "rb"))},
#     headers=headers
# )
# result: no error - but data is not recognized in miro

# TEST 2
# https://stackoverflow.com/questions/12385179/how-to-send-a-multipart-form-data-with-requests-in-python
# requests.post(
#     'http://requestb.in/xucj9exu',
#     files=(
#         ('foo', (None, 'bar')),
#         ('foo', (None, 'baz')),
#         ('spam', (None, 'eggs')),
#     )
# )

# response = requests.post(url=url,
#                          files=(
#                              ('resource', ("test.png", open(
#                                  r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb"))),
#                              ('data', (None, '"title for the image": "Test","position":{"x":100,"y":200,"origin":"center"},"geometry":{"width":100,"height":100,"rotation":0}"'))),
#                          headers=headers)
# result: no error - but data is not recognized in miro

# response = requests.post(url=url,
#                          files=(
#                              ('resource', ("test.png", open(
#                                  r"C:\Users\vbraz\Desktop\IMAGE_DATA_STICKY_NOTES\cando-dropbox-sticky-notes-images\only-yellow\IMG_8693.JPEG", "rb"))),
#                              ('data', (None, {"title for the image": "Test", "position": {"x": 100, "y": 200, "origin": "center"}, "geometry": {"width": 100, "height": 100, "rotation": 0}}))),
#                          headers=headers)
# result: error -> TypeError: a bytes-like object is required, not 'dict'

print(response)
print(response.text)
print("TEST")
