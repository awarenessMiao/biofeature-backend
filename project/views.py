from django.shortcuts import render
from django.http import HttpResponse
import json, base64, os, openface, cv2
import numpy as np

align = openface.AlignDlib('/root/openface/models/dlib/shape_predictor_68_face_landmarks.dat')
net = openface.TorchNeuralNet('/root/openface/models/openface/nn4.small2.v1.t7')

def encode_img(pic_img):
    retval, buffer = cv2.imencode('.jpg', pic_img)
    pic_str = base64.b64encode(buffer)
    return pic_str.decode()

def people(request):
    return HttpResponse(json.dumps(os.listdir('/data/lfw/')))

def photos(request):
    person = request.GET.get('name')
    return HttpResponse(json.dumps(os.listdir('/data/lfw/' + person)))

def image(request):
    person = request.GET.get('name')
    photo = request.GET.get('photo')
    image_data = open(os.path.realpath('/data/lfw/'+person+'/'+photo), 'rb').read()
    return HttpResponse(image_data, content_type='image/jpeg')

def decode_img(img_code):
    img_uri = img_code.split(",")[1]
    img_data = base64.b64decode(img_uri)
    img_array = np.fromstring(img_data, np.uint8)
    image = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    return image

def upload(request):
    info = json.loads(request.body)
    image = decode_img(info['img'])
    if info['person'] not in os.listdir('/data/lfw/'):
        os.mkdir(os.path.realpath('/data/lfw/'+info['person']), 0777)
    print('/data/lfw/'+info['person']+'/'+info['photo'])
    cv2.imwrite(os.path.realpath('/data/lfw/'+info['person']+'/'+info['photo']), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return HttpResponse(image, content_type='image/jpeg')

def test(request):
    person = request.GET.get('name')
    photo = request.GET.get('photo')
    image = cv2.imread(os.path.realpath('/data/lfw/'+person+'/'+photo), cv2.IMREAD_COLOR)
    rects = align.getAllFaceBoundingBoxes(image) # get green box(face bound)
    for rect in rects:  # for each face
        # draw green box
        cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0,255,0))
        shape = align.findLandmarks(image, rect) # find red points(eyes, mouth, nose, head)
        # print(len(shape))
        for pt in shape:
            # draw red points
            cv2.circle(image, pt, 3, [0, 0, 255], thickness=-1)
    cv2.imwrite('1.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    image_data = open('1.jpg', 'rb').read()
    return HttpResponse(image_data, content_type='image/jpeg')