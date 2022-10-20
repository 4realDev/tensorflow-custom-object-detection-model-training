import requests

url = "https://api.miro.com/v2/boards/uXjVO7Ddvh0%3D/images"

payload = {
    "data": {
        "url": "https://miro.com/static/images/page/mr-index/localization/en/slider/ideation_brainstorming.png",
        "title": "Test"
    },
    "position": {
        "origin": "center",
        "x": 0,
        "y": 0
    }
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer eyJtaXJvLm9yaWdpbiI6ImV1MDEifQ_N9OybOclP4WmwOKCNUjVuVMDshE"
}

response = requests.post(url, json=payload, headers=headers)
print(response)

print(response.text)
