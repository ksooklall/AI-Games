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

class Models:
    def __init__(self, width, height, output, learning_rate):
        self.width = width
        self.height = height
        self.output = output
        self.learning_rate = learning_rate
        
    def alexnet(self):
        network = input_data(shape=[None, self.width, self.height, 1], name='input')
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
        network = fully_connected(network, n_units=self.output, activation='softmax')
        network = regression(network, optimizer='momentum', loss='categorical_crossentropy',
                             learning_rate=self.learning_rate)
        mode = tflearn.DNN(network, checkpoint_path='alexnet',
                            max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
        return mode

    def alexnet_2(self):
        network = input_data(shape=[None, self.width, self.height, 1], name='input')
        network = conv_2d(network, nb_filter=96, filter_size=11, strides=4, activation='relu')
        network = max_pool_2d(network, kernel_size=3, strides=2)
        network = local_response_normalization(network)
        network = conv_2d(network, nb_filter=256, filter_size=5, strides=1, activation='relu')
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
        network = dropout(network, 0.6)
        network = fully_connected(network, n_units=4096, activation='tanh')
        network = dropout(network, 0.6)
        network = fully_connected(network, n_units=2048, activation='tanh')
        network = dropout(network, 0.6)
        network = fully_connected(network, n_units=self.output, activation='softmax')

        network = regression(network, optimizer='adam', loss='categorical_crossentropy',
                             learning_rate=self.learning_rate)
        model = tflearn.DNN(network, checkpoint_path='alexnet_2',
                            max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
        return model

    def vgg16(self):
        """
        Very Deep Convolutional Networks for Large-Scale Visual Recognition.
        VGG 16-layers convolutional with semantic segmentation
        References:
            Fully Convolutional Networks for Semantic Segmentation
            Jonathan Long*, Evan Shelhamer*, and Trevor Darrell. CVPR 2015.
        Links:
            https://arxiv.org/abs/1605.06211
        """
        x = tflearn.input_data(shape=[None, self.width, self.height, 1], name='input')

        x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
        x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

        x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
        x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

        x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
        x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
        x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
        x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
        x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

        x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
        x = tflearn.dropout(x, 0.5, name='dropout1')

        x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
        x = tflearn.dropout(x, 0.5, name='dropout2')

        x = tflearn.fully_connected(x, self.output, activation='softmax', scope='fc8')

        network = regression(x, optimizer='momentum', loss='categorical_crossentropy',
                             learning_rate=self.learning_rate)
        model = tflearn.DNN(network, checkpoint_path='vgg16',
                            max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
        return model
