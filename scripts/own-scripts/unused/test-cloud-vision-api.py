# pip install google-cloud-vision
# You can generate the credentials and the Json file by selecting your project on this link: console.cloud.google.com/iam-admin/serviceaccounts
from google.cloud import vision

client_options = {'api_endpoint': 'eu-vision.googleapis.com'}

client = vision.ImageAnnotatorClient(client_options=client_options)


def set_endpoint():
    """Change your endpoint"""
    # [START vision_set_endpoint]
    from google.cloud import vision

    client_options = {'api_endpoint': 'eu-vision.googleapis.com'}

    client = vision.ImageAnnotatorClient(client_options=client_options)
    # [END vision_set_endpoint]
    image_source = vision.ImageSource(
        image_uri='gs://cloud-samples-data/vision/text/screen.jpg')
    image = vision.Image(source=image_source)

    response = client.text_detection(image=image)

    print('Texts:')
    for text in response.text_annotations:
        print('{}'.format(text.description))

        vertices = ['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices]

        print('bounds: {}\n'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


if __name__ == '__main__':
    set_endpoint()
