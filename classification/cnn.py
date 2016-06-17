from __future__ import print_function

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

import sys
import os
import timeit

import numpy

from logReg import LogReg
from single_layer_MLP import HiddenLayer
from load import load, load_diff

class ConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
            numpy.prod(poolsize))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input

def train(learning_rate=0.01,
    n_epochs=10,
    dataset='mnist.pkl.gz',
    nkerns=[20, 50],
    batch_size=10,
    image_size=128,
    L1_reg = 0.0,
    L2_reg = 0.001):

    datasets = load()

    train_set_x, train_set_y = datasets[0]
    print(train_set_x)
    print(train_set_y)
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building the model')

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(123455)

    layer0_input = x.reshape((batch_size, 1, image_size, image_size))

    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, image_size, image_size),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    image_size_1 = (image_size - 4) / 2

    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], image_size_1, image_size_1),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    image_size_2 = (image_size_1 - 4) / 2
    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * image_size_2 * image_size_2,
        n_out=500,
        activation=T.tanh
    )

    layer3 = LogReg(
        input=layer2.output,
        n_in=500,
        n_out=3
    )

    L1 = (
        abs(layer0.W).sum()
        + abs(layer1.W).sum()
        + abs(layer2.W).sum()
        + abs(layer3.W).sum()
    )

    L2 = (
        abs(layer0.W).sum()
        + (layer1.W ** 2).sum()
        + (layer2.W ** 2).sum()
        + (layer3.W ** 2).sum()
    )

    cost = (layer3.negative_log_likelihood(y)
        + L1_reg * L1
        + L2_reg * L2
    )

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    params = layer3.params + layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    print('... training')

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()


    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print(minibatch_avg_cost)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                    in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100
                        )
                )

                if this_validation_loss < best_validation_loss:
                    if (
                        this_validation_loss < best_validation_loss * improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i
                        in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('    epoch %i, minibatch %i/%i, test error of '
                        'best model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                        test_score * 100.))

                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
        'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__=='__main__':
    train()
