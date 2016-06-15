from flask import Blueprint
from flask import request
from flask import abort
from PIL import Image
import json


api = Blueprint('api', __name__)

@api.route('/', methods = ['GET'])
def index():
    return 'Hello World'


@api.route('/', methods = ['POST'])
def stuff():
    print request
    print request.files['file']

    imgFile = Image.open(request.files['file'])

    imgFile.show()

    return 'cool beans', 200
