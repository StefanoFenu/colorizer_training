import os
from sklearn import metrics

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import skflow

from collections import namedtuple
from math import sqrt

def res_net(x, y, activation=tf.nn.relu):
    """
    Args:
    x: Input of the network
    y: Output of the network
    activation: Activation function to apply after each convolution
    """
    print 'Initializing net'
    # Configurations for each bottleneck block
    BottleneckBlock = namedtuple(
    'BottleneckBlock', ['num_layers', 'num_filters', 'bottleneck_size'])
    blocks = [BottleneckBlock(3, 128, 32),
    BottleneckBlock(3, 256, 64),
    BottleneckBlock(3, 512, 128),
    BottleneckBlock(3, 1024, 256)]

    input_shape = x.get_shape().as_list()

    print x
    print input_shape
    # Reshape the input into the right shape if it's 2D tensor
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    # First convolution expands to 64 channels
    with tf.variable_scope('conv_layer1'):
        net = skflow.ops.conv2d(x, 64, [7, 7], batch_norm=True,
        activation=activation, bias=False)

        # Max pool
        net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # First chain of resnets
    with tf.variable_scope('conv_layer2'):
        net = skflow.ops.conv2d(net, blocks[0].num_filters,
           [1, 1], [1, 1, 1, 1], padding='VALID', bias=True)

    # Create each bottleneck building block for each layer
    for block_i, block in enumerate(blocks):
        for layer_i in range(block.num_layers):

            name = 'block_%d/layer_%d' % (block_i, layer_i)
            print name
            # 1x1 convolution responsible for reducing dimension
            with tf.variable_scope(name + '/conv_in'):
                conv = skflow.ops.conv2d(net, block.num_filters,
                [1, 1], [1, 1, 1, 1],
                padding='VALID',
                activation=activation,
                batch_norm=True,
                bias=False)

            with tf.variable_scope(name + '/conv_bottleneck'):
                conv = skflow.ops.conv2d(conv, block.bottleneck_size,
                [3, 3], [1, 1, 1, 1],
                padding='SAME',
                activation=activation,
                batch_norm=True,
                bias=False)

            # 1x1 convolution responsible for restoring dimension
            with tf.variable_scope(name + '/conv_out'):
                conv = skflow.ops.conv2d(conv, block.num_filters,
                [1, 1], [1, 1, 1, 1],
                padding='VALID',
                activation=activation,
                batch_norm=True,
                bias=False)

                # shortcut connections that turn the network into its counterpart
                # residual function (identity shortcut)
                net = conv + net

        try:
            # upscale to the next block size
            next_block = blocks[block_i + 1]
            with tf.variable_scope('block_%d/conv_upscale' % block_i):
                net = skflow.ops.conv2d(net, next_block.num_filters,
                [1, 1], [1, 1, 1, 1],
                bias=False,
                padding='SAME')
        except IndexError:
            pass

    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(net,
    ksize=[1, net_shape[1], net_shape[2], 1],
    strides=[1, 1, 1, 1], padding='VALID')

    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    foo = skflow.models.logistic_regression(net, y)


    return foo 


if __name__ == '__main__':
    from colorizer_models import load_data

    print 'Loading data'
    x_train, y_train, x_test, y_test = load_data()

    print 'Initializing classifier'

    classifier = skflow.TensorFlowEstimator(
    model_fn=res_net, n_classes=256*3, batch_size=128, verbose=1,
    steps=100, learning_rate=0.001, continue_training=True)

    print 'Fitting classifier'
    # Train model and save summaries into logdir.

    classifier.fit(x_train, y_train, logdir="models/resnet/")

    # Calculate accuracy.
    score = metrics.accuracy_score(
    y_test, classifier.predict(x_test, batch_size=64))
    print('Accuracy: {0:f}'.format(score))

    # Save model graph and checkpoints.
    classifier.save("models/resnet/")
