from flask import Blueprint
from flask import request
from flask import abort
from flask import jsonify
from PIL import Image
import json
import sys

sys.path.insert(0, 'classifier')

from predict import predict


api = Blueprint('api', __name__)

@api.route('/', methods = ['GET'])
def index():
    return 'Hello World'


@api.route('/', methods = ['POST'])
def stuff():
    print request
    if not request.files:
        abort(400)

    print request.files['file']

    img = Image.open(request.files['file'])

    p = predict(img)

    return jsonify(plantnName='Red Oak',
                   species='Acer rubrum',
                   imgUrl='https://en.wikipedia.org/wiki/Acer_rubrum#/media/File:2014-10-30_11_09_40_Red_Maple_during_autumn_on_Lower_Ferry_Road_in_Ewing,_New_Jersey.JPG'
                   )
