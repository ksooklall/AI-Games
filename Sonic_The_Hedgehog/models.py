"""
Models for training AI
"""
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

"""
Alex net
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""

def alexnet(width, height, output, learning_rate):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, nb_filter=96, filter_size=11, strides=4, activation='relu')
    network = max_pool_2d(network, kernel_size=3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, nb_filter=256, filter_size=5, strides=1, activation='relu')
    network = max_pool_2d(network, kernel_size=3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, nb_filter=384, filter_size=3, strides=1, activation='relu')
    network = conv_2d(network, nb_filter=384, filter_size=3, strides=1, activation='relu')
    network = conv_2d(network, nb_filter=256, filter_size=3, strides=1, activation='relu')
    network = max_pool_2d(network, kernel_size=3, strides=2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, n_units=4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, n_units=4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, n_units=output, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=learning_rate)
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model
