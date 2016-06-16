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
    file = request.files['file']
    if file:
         img = Image.open(file)
         img.show()
    return 'sup', 200
