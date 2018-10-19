#!/usr/bin/env python

import sys
import os
sys.path.append('..')
from blocks.depth_completion_experiment import DepthCompletionExperiment
from blocks.sparse_conv import sparse_conv

import tensorflow as tf

class SparseConvExperiment(DepthCompletionExperiment):
    def __init__(self):
        super(SparseConvExperiment, self).__init__()
        self.parameters.dataset_train['input'] = os.path.join("..","datasets", "sparse_train.dataset")
        self.parameters.dataset_train['label'] = os.path.join("..","datasets", "dense_train.dataset")

        self.parameters.dataset_val['input'] = os.path.join("..","datasets", "sparse_val.dataset")
        self.parameters.dataset_val['label'] = os.path.join("..","datasets", "dense_val.dataset")

        self.parameters.image_size = (352, 1216)

        self.parameters.steps_per_epoch = 85896
        self.parameters.max_epochs=10
        self.parameters.batchsize = 1
        self.parameters.steps_per_epoch = self.parameters.steps_per_epoch / \
            self.parameters.batchsize

        self.parameters.num_steps = self.parameters.steps_per_epoch * self.parameters.max_epochs

        self.parameters.log_dir = '../logs/sparse_conv_net/'

        self.parameters.l2_scale = 2.0
        self.parameters.learning_rate = 0.001
        #self.parameters.optimizer = tf.train.MomentumOptimizer(
        #    learning_rate=self.parameters.learning_rate, 
        #    momentum=0.9,
        #    use_nesterov=True)
        self.parameters.optimizer = tf.train.AdamOptimizer(
             learning_rate=self.parameters.learning_rate)
        self.parameters.loss_function = tf.losses.mean_squared_error

    def network(self, tf_input, **kwargs):

        reuse = kwargs.get('reuse', False)

        with tf.variable_scope('SparseConvNet', reuse=reuse):
            b,h,w,c = tf_input.get_shape()
            print tf_input
            channels=tf.split(tf_input,c,axis=3)
            #assume that if one channel has no information, ALL CHANNELS HAVE NO INFORMATION
            mask = tf.where(tf.equal(channels[0], 0), 
                            tf.zeros_like(channels[0]), 
                            tf.ones_like(channels[0])) #mask should only have the size of (B,H,W,1)

            x, mask = sparse_conv(tf_input, mask=mask, filters=16, kernel_size=(11, 11), l2_scale=self.parameters.l2_scale, padding='same', name='sparse_conv1')
            x = tf.nn.relu(x)
            x, mask = sparse_conv(x, mask=mask, filters=16, kernel_size=(7, 7), l2_scale=self.parameters.l2_scale, padding='same', name='sparse_conv2')
            x = tf.nn.relu(x)
            x, mask = sparse_conv(x, mask=mask, filters=16, kernel_size=(5, 5), l2_scale=self.parameters.l2_scale, padding='same', name='sparse_conv3')
            x = tf.nn.relu(x)
            x, mask = sparse_conv(x, mask=mask, filters=16, kernel_size=(3, 3), l2_scale=self.parameters.l2_scale, padding='same', name='sparse_conv4')
            x = tf.nn.relu(x)
            x, mask = sparse_conv(x, mask=mask, filters=16, kernel_size=(3, 3), l2_scale=self.parameters.l2_scale, padding='same', name='sparse_conv5')
            x = tf.nn.relu(x)
            x, mask = sparse_conv(x, mask=mask, filters=1, kernel_size=(1, 1), l2_scale=self.parameters.l2_scale, padding='same', name='sparse_conv6')

            return x*mask

if __name__ == '__main__':
    exp = SparseConvExperiment()
    exp.run()