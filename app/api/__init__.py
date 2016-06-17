from flask import Blueprint
from flask import request
from flask import abort
from flask import jsonify
from PIL import Image
import os
import json
import sys

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

    img = request.files['file']


    p = predict(img)
    print p

    img = Image.open(request.files['file'])

    return json.dumps([{'plantName':'Red Oak',
                   'imgUrl':'http://cossdotblog.wpengine.netdna-cdn.com/wp-content/uploads/2010/05/red_maple_fall-RESIZED.jpg',
                   'bloomTime':'summer!', 'matchPercent': '99.9%'}])
