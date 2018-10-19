'''This module provides visualizations for depth images.
'''

import numpy as np
import tensorflow as tf

# the false color map
color_map_bins = np.array([114.0, 185.0, 114.0, 174.0, 114.0, 185.0, 114.0, 0.0], dtype=np.float32)

color_map_valids = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], 
                             [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=np.float32)

# helper for fast calculation
sum = np.sum(color_map_bins)
weights = sum / color_map_bins[:7]
color_map_bins[0] = 0
cumsum = np.cumsum(color_map_bins/sum)
    
def depth_colored(tf_input, max_val=70):

    with tf.name_scope('visualization') as visualization:
        normed = tf.minimum(tf.maximum(tf_input / max_val, 0.0), 1.0)
        # duplicate the channels to substract the cumsum later
        normed_7 = tf.tile(normed, [1,1,1,7])
        diff = normed_7 - cumsum[:7]
        # get the index of the cumsum by getting the min of the difference
        idx_image = tf.argmin(tf.abs(diff), axis=-1, output_type=tf.int32)
        idx_image = tf.expand_dims(idx_image, axis=-1)

        # create images that take the value of the cumsum at the index value
        cumsum_image = tf.cast(tf.gather_nd(params=cumsum, indices=idx_image), tf.float32)
        cumsum_image = tf.expand_dims(cumsum_image, axis=-1)
        # create images that take the value of the weight at the index value
        weight_image = tf.cast(tf.gather_nd(params=weights, indices=idx_image), tf.float32)
        weight_image = tf.expand_dims(weight_image, axis=-1)
        # create images that take the rgb masks at the index value
        color_map_image = tf.cast(tf.gather_nd(params=color_map_valids, indices=idx_image), tf.float32)
        # create images that take the rgb masks at the next index value
        color_map_image_next_bin = tf.cast(tf.gather_nd(params=color_map_valids, indices=idx_image+1), tf.float32)

        # calculate the weight
        w = 1.0 - (normed - cumsum_image) * weight_image
        colored_image = (w * color_map_image + (1.0 - w) * color_map_image_next_bin) * 255.0
        colored_image = tf.cast(colored_image, tf.uint8)

        return colored_image
