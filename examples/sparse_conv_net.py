#!/usr/bin/env python

import sys
import os
sys.path.append('..')
from blocks.depth_completion_experiment import DepthCompletionExperiment
from blocks.sparse_conv import SparseConv
from keras.layers import Activation

from keras import optimizers
import tensorflow as tf

class SparseConvExperiment(DepthCompletionExperiment):
    def __init__(self):
        super(SparseConvExperiment, self).__init__()
        self.parameters.dataset_train['input'] = os.path.join("..","datasets", "sparse_train.dataset")
        self.parameters.dataset_train['label'] = os.path.join("..","datasets", "dense_train.dataset")

        self.parameters.dataset_val['input'] = os.path.join("..","datasets", "sparse_val.dataset")
        self.parameters.dataset_val['label'] = os.path.join("..","datasets", "dense_val.dataset")

        self.parameters.image_size = (1216, 352)

        self.parameters.steps_per_epoch = 85896
        self.parameters.batchsize = 1
        self.parameters.steps_per_epoch = self.parameters.steps_per_epoch / \
            self.parameters.batchsize

        self.parameters.learning_rate = 0.01
        self.parameters.optimizer = optimizers.Adagrad(
            lr=self.parameters.learning_rate, epsilon=None, decay=0.0)
        self.parameters.loss_function = 'mean_squared_error'

    def network(self, tf_input):

        b,h,w,c = tf_input.get_shape()
        channels=tf.split(tf_input,c,axis=3)
        #assume that if one channel has no information, ALL CHANNELS HAVE NO INFORMATION
        mask = tf.where(tf.equal(channels[0], 0), 
                        tf.zeros_like(channels[0]), 
                        tf.ones_like(channels[0])) #mask should only have the size of (B,H,W,1)

        x, mask = SparseConv(mask=mask, filters=16, kernel_size=(11, 11), padding='same')(tf_input)
        x = Activation('relu')(x)
        x, mask = SparseConv(mask=mask, filters=16, kernel_size=(7, 7), padding='same')(x)
        x = Activation('relu')(x)
        x, mask = SparseConv(mask=mask, filters=16, kernel_size=(5, 5), padding='same')(x)
        x = Activation('relu')(x)
        x, mask = SparseConv(mask=mask, filters=16, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x, mask = SparseConv(mask=mask, filters=16, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x, mask = SparseConv(mask=mask, filters=1, kernel_size=(1, 1), padding='same')(x)

        return x

if __name__ == '__main__':
    exp = SparseConvExperiment()
    exp.run()