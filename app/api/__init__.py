from flask import Blueprint
import json


api = Blueprint('api', __name__)

@api.route('/')
def index():
    return 'Hello World'


@api.route('/stuff')
def stuff(stuff):
    print stuff
    f = open('stuff', 'w')
    f.write(stuff)
    f.close()
    return stuff
