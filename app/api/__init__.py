from flask import Blueprint
from flask import request
from flask import abort
import json


api = Blueprint('api', __name__)

@api.route('/')
def index():
    return 'Hello World'


@api.route('/stuff', methods = ['POST'])
def stuff():
    print request.json
    if not request.json or not 'title' in request.json:
        print 'hey'
        abort(400)

    print request
    # f = open('stuff', 'w')
    # f.write(stuff)
    # f.close()
    return 'sup', 200
