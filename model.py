#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


class Model:
    def __init__(self):
        self.weights = {}
        self.biases  = {}

    def logit(self, data, isTrain=True, dropout=0.7):
        print('[model] data shape = {0}'.format(data.shape))

        with tf.variable_scope('conv1') as scope:
            data = self.conv(data, 3, 8, [1, 1, 1, 1], scope.name + '_1')
            data = self.conv(data, 3, 8, [1, 1, 1, 1], scope.name + '_2')
            data = tf.nn.max_pool(data, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            data = self.dropout(data, isTrain, dropout)
            print('[model] data shape at conv1 = {0}'.format(data.shape))

        with tf.variable_scope('conv2') as scope:
            data = self.conv(data, 3, 16, [1, 1, 1, 1], scope.name + '_1')
            data = self.conv(data, 3, 16, [1, 1, 1, 1], scope.name + '_2')
            data = tf.nn.max_pool(data, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
            data = self.dropout(data, isTrain, dropout)
            print('[model] data shape at conv2 = {0}'.format(data.shape))

        with tf.variable_scope('fc1') as scope:
            shape = data.get_shape().as_list()
            dim = np.prod(shape[1:])
            data = tf.reshape(data, [-1, dim])

            data = self.fc(data, 256, scope.name)
            data = self.dropout(data, isTrain, dropout)
            print('[model] data shape at fc1 = {0}'.format(data.shape))

        with tf.variable_scope('fc2') as scope:
            data = self.fc(data, 10, scope.name)
            data = self.dropout(data, isTrain, dropout)
            print('[model] data shape at fc2 = {0}'.format(data.shape))

        return data

    def conv(self, data, kernel_size, out_channel, stride, name, isActivate=True):
        in_channel = data.get_shape().as_list()[3]
        kernel_shape = [kernel_size, kernel_size, in_channel, out_channel]

        weights_name = name + '_kernel'
        biases_name = name + '_biases'

        weights = self.get_weights_var(weights_name, kernel_shape)
        biases = self.get_biases_var(biases_name, out_channel)

        self.set_weights(weights_name, weights)
        self.set_biases(biases_name, biases)
        
        data = tf.nn.conv2d(data, weights, stride, 'SAME')
        data = tf.nn.bias_add(data, biases)
        if isActivate:
            data = tf.nn.relu(data)
        
        return data
    
    def fc(self, data, num_output, name):
        shape = data.get_shape().as_list()
        num_input = shape[1]
        
        weights_name = name + '_kernel'
        biases_name = name + '_biases'
        weights_shape = [num_input, num_output]

        weights = self.get_weights_var(weights_name, weights_shape)
        biases = self.get_biases_var(biases_name, num_output)

        self.set_weights(weights_name, weights)
        self.set_biases(biases_name, biases)

        data = tf.matmul(data, weights)
        data = tf.nn.bias_add(data, biases)
        
        return data

    def dropout(self, data, isTrain, dropout):
        if isTrain:
            if (dropout > 0.0) and (dropout < 1.0):
                data = tf.nn.dropout(data, dropout)
        return data

    def get_weights_var(self, weights_name, weights_shape):
        init = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
        return tf.get_variable(weights_name, weights_shape, initializer=init)

    def get_biases_var(self, biases_name, num_output):
        init = tf.constant_initializer(value=0, dtype=tf.float32)
        return tf.get_variable(biases_name, [num_output], initializer=init)

    def set_weights(self, weights_name, weights):
        self.weights[weights_name] = weights

    def set_biases(self, biases_name, biases):
        self.biases[biases_name] = biases

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases
