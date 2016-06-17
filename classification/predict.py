from __future__ import print_function

import six.moves.cPickle as pickle
import dill
import gzip
import os
import sys
import timeit

from PIL import Image

import numpy

import theano
import theano.tensor as T

from load import load

def predict(img):
    try:
        classifier = dill.load(open('best_model.pkl'))
    except:
        classifier = dill.load(open('../../../best_model.pkl'))

    img = Image.open(img).convert('L')
    img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
    img = img.resize((128, 128))
    img = list(img.getdata())
    img = numpy.asarray(img)

    predict_model = theano.function(
        inputs=[classifier.x],
        outputs=classifier.y_pred,
    )

    predicted_values = predict_model([img] * 10)
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

    return predicted_values

if __name__=='__main__':
    predict(sys.argv[1])
