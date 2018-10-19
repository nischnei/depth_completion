#!/usr/bin/env python

import tensorflow as tf
import sys
import os
sys.path.append('..')
from blocks.depth_completion_experiment import DepthCompletionExperiment

class StdConvExperiment(DepthCompletionExperiment):

    def __init__(self):
        super(StdConvExperiment, self).__init__()
        self.parameters.dataset_train['input'] = os.path.join(
            "..", "datasets", "sparse_train.dataset")
        self.parameters.dataset_train['label'] = os.path.join(
            "..", "datasets", "dense_train.dataset")

        self.parameters.dataset_val['input'] = os.path.join(
            "..", "datasets", "sparse_val.dataset")
        self.parameters.dataset_val['label'] = os.path.join(
            "..", "datasets", "dense_val.dataset")

        self.parameters.image_size = (352, 1216)

        self.parameters.steps_per_epoch = 85896
        self.parameters.max_epochs=10
        self.parameters.batchsize = 1
        self.parameters.steps_per_epoch = self.parameters.steps_per_epoch / \
            self.parameters.batchsize

        self.parameters.l2_scale = 2.0

        self.parameters.num_steps = self.parameters.steps_per_epoch * self.parameters.max_epochs

        self.parameters.log_dir = '../logs/std_conv_net/'

        self.parameters.learning_rate = 0.001
        self.parameters.optimizer = tf.train.AdamOptimizer(
             learning_rate=self.parameters.learning_rate)
        self.parameters.loss_function = tf.losses.mean_squared_error

    def network(self, tf_input, **kwargs):

        reuse = kwargs.get('reuse', False)

        with tf.variable_scope('StdConvNet', reuse=reuse):
            initializer = tf.contrib.layers.xavier_initializer()
            x = tf.layers.conv2d(tf_input, filters=16, kernel_size=(11, 11), kernel_initializer=initializer,
                                strides=(1, 1), trainable=True, padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.parameters.l2_scale), name="conv1")
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=16, kernel_size=(7, 7), kernel_initializer=initializer,
                                strides=(1, 1), trainable=True, padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.parameters.l2_scale), name="conv2")
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=16, kernel_size=(5, 5), kernel_initializer=initializer,
                                strides=(1, 1), trainable=True, padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.parameters.l2_scale), name="conv3")
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=16, kernel_size=(3, 3), kernel_initializer=initializer,
                                strides=(1, 1), trainable=True, padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.parameters.l2_scale), name="conv4")
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=16, kernel_size=(3, 3), kernel_initializer=initializer,
                                strides=(1, 1), trainable=True, padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.parameters.l2_scale), name="conv5")
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=1, kernel_size=(1, 1), kernel_initializer=initializer,
                                strides=(1, 1), trainable=True, padding='same', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.parameters.l2_scale), name="conv6")

            return x


if __name__ == '__main__':
    exp = StdConvExperiment()
    exp.run()
