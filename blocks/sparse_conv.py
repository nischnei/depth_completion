import tensorflow as tf

# Code implemented by https://github.com/PeterTor/sparse_convolution

"""Arguments
   tensor: Tensor input.
   binary_mask: Tensor, a mask with the same size as tensor, channel size = 1
   filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
   kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
   strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
   l2_scale: float, A scalar multiplier Tensor. 0.0 disables the regularizer.
   
 Returns:
   Output tensor, binary mask.
 """


def sparse_conv(tensor, mask=None,
                filters=32,
                kernel_size=3,
                strides=1,
                l2_scale=2.0,
                padding='same',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='sparse_conv'):

    with tf.name_scope(name) as sparse_conv_scope:
        if mask == None:  # first layer has no binary mask
            b, h, w, c = tensor.get_shape()
            channels = tf.split(tensor, c, axis=3)
            # assume that if one channel has no information, all channels have no information
            mask = tf.where(tf.equal(channels[0], 0),
                            tf.zeros_like(channels[0]),
                            tf.ones_like(channels[0]))  # mask should only have the size of (B,H,W,1)

        features = tf.multiply(tensor, mask)
        features = tf.layers.conv2d(features,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    kernel_initializer=kernel_initializer,
                                    strides=(strides, strides),
                                    trainable=True,
                                    use_bias=False,
                                    padding=padding,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        scale=l2_scale),
                                    name=name+"_conv")

        norm = tf.layers.conv2d(mask,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=(strides, strides),
                                kernel_initializer=tf.ones_initializer(),
                                trainable=False,
                                use_bias=False,
                                padding=padding,
                                name=name+"_norm")

        norm = tf.where(tf.equal(norm, 0), tf.zeros_like(
            norm), tf.reciprocal(norm))
        _, _, _, bias_size = norm.get_shape()

        # add a bias term
        b = tf.Variable(tf.constant(
            0.0, shape=[bias_size]), trainable=True, name=name+"_bias")

        feature = tf.multiply(features, norm)+b
        mask = tf.layers.max_pooling2d(
            mask, strides=strides, pool_size=kernel_size, padding=padding, name=name+"_mask_out")

        return feature, mask
