from flask import Blueprint
from flask import request
from flask import abort
import json


api = Blueprint('api', __name__)

@api.route('/', methods = ['GET'])
def index():
    return 'Hello World'


@api.route('/', methods = ['POST'])
def stuff():
    print request
    print type(request)
    
    # if not request.json in request.json:
    #     print 'fail'
    #     abort(400)

    # f = open('stuff', 'w')
    # f.write(stuff)
    # f.close()
    return 'sup', 200
