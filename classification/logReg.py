from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class LogReg(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )

        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        print('errors')
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            print(y.shape)
            print(self.y_pred.shape)
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()