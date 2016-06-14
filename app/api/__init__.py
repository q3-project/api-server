from flask import Blueprint


api = Blueprint('api', __name__)

@api.route('/')
def index(stuff):
    print stuff
    f = open('stuff', 'w')
    f.write(stuff)
    f.close()
    return stuff
