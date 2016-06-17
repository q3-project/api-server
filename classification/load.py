from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys

import numpy

import theano
import theano.tensor as T

from PIL import Image


def load():
    species_list = [
        'acer_rubrum',
        'ptelea_trifoliata',
        'ulmus_rubra'
        # 'test1',
        # 'test2',
        # 'test3'
    ]
    STANDARD_SIZE = (128, 128)

    def load_leaf_train(species, tag):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'lab',
            species
        )
        print(path)


        directory = os.listdir(path)
        images = []
        for file in directory:
            if file != '.DS_Store':
                img = Image.open(path + '/' + file).convert('L')
                img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
                img = img.resize(STANDARD_SIZE)
                img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
                img2 = list(img2.getdata())
                img = list(img.getdata())
                # img = map(list, img)
                img = numpy.array(img)
                img2 = numpy.array(img2)
                images.append((img, tag))
                images.append((img2, tag))

        return images

    def load_leaf_test(species, tag):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'field',
            species
        )
        print(path)

        directory = os.listdir(path)
        images = []

        i = 0

        for file in directory:
            i += 1
            if file != '.DS_Store':
                if i % 2 == 0:
                    img = Image.open(path + '/' + file).convert('L')
                    img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
                    img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append((img, tag))

        return images

    def load_leaf_valid(species, tag):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'field',
            species
        )
        print(path)

        directory = os.listdir(path)
        images = []

        i = 0

        for file in directory:
            i += 1
            if file != '.DS_Store':
                if i % 2 == 1:
                    img = Image.open(path + '/' + file).convert('L')
                    img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
                    img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append((img, tag))

        return images

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = zip(*data_xy)
        data_x = list(data_x)
        data_x = numpy.asarray(data_x)
        data_y = list(data_y)
        data_y = numpy.asarray(data_y)

        shared_x = theano.shared(numpy.asarray(data_x,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )
        shared_y = theano.shared(numpy.asarray(data_y,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )

        return shared_x, T.cast(shared_y, 'int32')

    train = []
    valid = []
    test = []

    tag = 0
    for species in species_list:
        print('... loading %s, as %i' % (species, tag))
        train += load_leaf_train(species, tag)
        valid += load_leaf_valid(species, tag)
        test += load_leaf_test(species, tag)

        tag += 1

    train = numpy.asarray(train)
    valid = numpy.asarray(valid)
    test = numpy.asarray(test)

    numpy.random.shuffle(train)

    numpy.random.shuffle(valid)

    numpy.random.shuffle(test)

    train_x, train_y = shared_dataset(train)
    test_x, test_y = shared_dataset(test)
    valid_x, valid_y = shared_dataset(valid)

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

    return rval



def load_diff():
    species_list = [
        'acer_rubrum',
        'ptelea_trifoliata',
        'ulmus_rubra'
        # 'test1',
        # 'test2',
        # 'test3'
    ]
    STANDARD_SIZE = (128, 128)

    def load_leaf_train(species, tag):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'lab',
            species
        )
        print(path)

        i = 1
        directory = os.listdir(path)
        images = []
        for file in directory:
            if file != '.DS_Store':
                i += 1
                if i % 3 == 0:
                    img = Image.open(path + '/' + file).convert('L')
                    img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
                    img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append((img, tag))

        return images

    def load_leaf_test(species, tag):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'lab',
            species
        )
        print(path)

        directory = os.listdir(path)
        images = []

        i = 2

        for file in directory:
            if file != '.DS_Store':
                i += 1
                if i % 3 == 0:
                    img = Image.open(path + '/' + file).convert('L')
                    img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
                    img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append((img, tag))

        return images

    def load_leaf_valid(species, tag):
        path = os.path.join(
            os.path.split(__file__)[0],
            'data',
            'leafsnap-dataset',
            'dataset',
            'images',
            'lab',
            species
        )
        print(path)

        directory = os.listdir(path)
        images = []

        i = 3

        for file in directory:
            if file != '.DS_Store':
                i += 1
                if i % 3 == 0:
                    img = Image.open(path + '/' + file).convert('L')
                    img = img.crop((150, 150, img.size[0] - 150, img.size[1] - 150))
                    img = img.resize(STANDARD_SIZE)
                    img = list(img.getdata())
                    # img = map(list, img)
                    img = numpy.array(img)
                    images.append((img, tag))

        return images

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = zip(*data_xy)
        data_x = list(data_x)
        data_x = numpy.asarray(data_x)
        data_y = list(data_y)
        data_y = numpy.asarray(data_y)

        shared_x = theano.shared(numpy.asarray(data_x,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )
        shared_y = theano.shared(numpy.asarray(data_y,
                dtype=theano.config.floatX
            ),
            borrow=borrow
        )

        return shared_x, T.cast(shared_y, 'int32')

    train = []
    valid = []
    test = []

    tag = 0
    for species in species_list:
        print('... loading %s, as %i' % (species, tag))
        train += load_leaf_train(species, tag)
        valid += load_leaf_valid(species, tag)
        test += load_leaf_test(species, tag)

        tag += 1

    train = numpy.asarray(train)
    valid = numpy.asarray(valid)
    test = numpy.asarray(test)

    numpy.random.shuffle(train)

    numpy.random.shuffle(valid)

    numpy.random.shuffle(test)

    train_x, train_y = shared_dataset(train)
    test_x, test_y = shared_dataset(test)
    valid_x, valid_y = shared_dataset(valid)

    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]

    return rval
