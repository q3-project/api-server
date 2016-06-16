from flask import Blueprint
from flask import request
from flask import abort
from flask import jsonify
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

    return jsonify(plantnName='Chinkapin Oak',
                   species='Quercus muehlenbergii',
                   imgUrl='http://www.nature.org/ourinitiatives/regions/northamerica/unitedstates/tennessee/chinkapin-oak-leaf-640x400.jpg',
                   habitat='Dry, rocky soils', growthHabit='Deciduous tree, growing 15-25 m tall', bloomTime='Mid-spring', longevity='Long-lived'
                   )
