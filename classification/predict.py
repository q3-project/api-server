from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


def predict(img):
    classifier = pickle.load(open('best_model.pkl'))

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    predicted_values = predict_model(img)
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
