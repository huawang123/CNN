import tensorflow as tf
from utils import conv2layer, pool2layer
from tensorflow.contrib.framework import arg_scope

def vgg16(images, n_classes, keep_prob, trainable=False, reuse=None):
    with tf.variable_scope("vgg16", reuse=reuse) as sc:
        with arg_scope([conv2layer,pool2layer], stride=1, padding_mode='SAME'):
            with arg_scope([conv2layer], activation='relu', bn=True, trainning=trainable):
                net = conv2layer(images=images, kernel=[3, 3], output_channel=64,scope='conv1')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=64, scope='conv2')

                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling1')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=128, scope='conv3')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=128, scope='conv4')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling2')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=256, scope='conv5')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=256, scope='conv6')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=256, scope='conv7')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling3')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv8')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv9')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv10')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling4')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv11')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv12')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv13')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling5')

                net = conv2layer(images=net, kernel=[8, 8], output_channel=256, padding_mode='VALID', scope='fc1')
                net = tf.nn.dropout(net, keep_prob=keep_prob)
                net = conv2layer(images=net, kernel=[1, 1], output_channel=128, scope='fc2')
                net = tf.nn.dropout(net, keep_prob=keep_prob)

                net = conv2layer(images=net, kernel=[1, 1], output_channel=n_classes, activation=None, scope='logits')
                net = tf.squeeze(net, axis=[1, 2], name='squeeze_logits')

                return net
