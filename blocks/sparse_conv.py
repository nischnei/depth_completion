from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.base_layers import InputSpec

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
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
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
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)


        self.kernel_ones = self.add_weight(shape=kernel_shape,
                                           initializer=initializers.Ones(),
                                           name='kernel',
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
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # multiply the inputs with the mask first, so no invalid entries exist
        features = K.multiply(inputs, self.mask)

        # do convolution on features with trainable weights
        features = K.conv2d(
                features,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                use_bias=False)

        # calculate the normalization
        norm = K.conv2d(
                self.mask,
                self.kernel_ones,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                use_bias=False)

        norm = tf.where(tf.equal(norm,0),tf.zeros_like(norm),tf.reciprocal(norm))
        _,_,_,bias_size = norm.get_shape()

        feature = K.multiply(features,norm)

        if self.use_bias:
            feature = K.bias_add(
                feature,
                self.bias,
                data_format=self.data_format)

        newMask = K.MaxPooling2D(self.mask,
            strides=self.strides,
            pool_size=self.kernel_size)

        if self.activation is not None:
            return self.activation(feature)
        return feature, newMask

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
        return [(input_shape[0], self.filters) + tuple(new_space)]*2
        