from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import InputSpec

from keras.utils import conv_utils

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

import tensorflow as tf


class SparseConv(Layer):

    def __init__(self,
                 mask,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SparseConv, self).__init__(**kwargs)
        self.mask = mask
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        print self.data_format
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        kernel_shape_ones = self.kernel_size + (1, 1)
        print kernel_shape

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_ones = self.add_weight(shape=kernel_shape_ones,
                                           initializer=initializers.Ones(),
                                           name='kernel_ones',
                                           regularizer=None,
                                           constraint=None,
                                           trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # multiply the inputs with the mask first, so no invalid entries exist
        features = inputs * self.mask
        print(features)

        # do convolution on features with trainable weights
        features = K.conv2d(
            features,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format)

        # calculate the normalization
        norm = K.conv2d(
            self.mask,
            self.kernel_ones,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format)

        norm = tf.where(tf.equal(norm, 0), tf.zeros_like(
            norm), tf.reciprocal(norm))
        _, _, _, bias_size = norm.get_shape()

        feature = features * norm

        if self.use_bias:
            feature = K.bias_add(
                feature,
                self.bias,
                data_format=self.data_format)

        self.newMask = K.pool2d(self.mask,
                                strides=self.strides,
                                pool_size=self.kernel_size,
                                padding='same',
                                data_format=self.data_format,
                                pool_mode='max')

        if self.activation is not None:
            self.feature = self.activation(feature)
        return [self.feature, self.newMask]

    def compute_output_shape(self, input_shape):
        print "SHAPPEE"
        print(tuple(K.int_shape(self.newMask)))
        return [tuple(K.int_shape(self.feature)),
                tuple(K.int_shape(self.newMask))]

    def compute_mask(self, input, input_mask=None):
        return [None, None]
