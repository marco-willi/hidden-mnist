import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np


class IdentityHalfLayer(tf.keras.layers.Wrapper):
    """ Wrapper that applies the Identity transformation to 
        one half of the batch and applies Layer to the other
        Only valid if Layer output shape == input shape
    """
    def __init__(self,  layer, **kwargs):
        super(IdentityHalfLayer, self).__init__(layer, **kwargs)

    def __call__(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=0)
        a = self.layer(a, **kwargs)
        return tf.concat([a, b], axis=0)


class GaussianBlurr2D(tf.keras.initializers.Initializer):
    """ Initializer that generates a 2D Gaussian Kernel
        Args: 
            sigma: variance of the Gaussian
            prob: probability of the layer being applied on a sample
    """

    def __init__(self, sigma, prob=1):
        if not sigma > 0:
            raise ValueError("'sigma' must be positive")

        self.sigma = tf.cast(sigma, tf.float32)
        self.dist = tfp.distributions.Normal(loc=0, scale=sigma)

    def __call__(self, shape, dtype=None):

        if len(shape) != 4:
            raise ValueError("'shape' must be 4 dimensional")
        
        if shape[0] != shape[1]:
            raise ValueError("'shape' must be square")

        if shape[0] < 3:
            raise ValueError("'shape' must be at least 5")

        if (shape[0] % 2) != 1:
            raise ValueError("'shape' must be an odd number")

        kernel_radius = shape[0] // 2

        probs = self.dist.prob(
            tf.range(-kernel_radius, kernel_radius + 1, 1, dtype=tf.float32))

        kernel_vals = tf.tile(probs, tf.expand_dims(probs.shape[0], -1))

        kernel_vals_2D = tf.reshape(
            kernel_vals, shape=(probs.shape[0], probs.shape[0]))

        kernel = tf.multiply(kernel_vals_2D, tf.transpose(kernel_vals_2D))

        assert kernel.shape[0:2] == shape[0:2], \
         "kernel shape does not match requested shape"

        kernel = kernel / tf.reduce_sum(kernel)

        # replicate kernel across channels
        dim = 2
        while len(kernel.shape) < len(shape):
            kernel = tf.expand_dims(kernel, -1)
            kernel = tf.keras.backend.repeat_elements(
                kernel, rep=shape[dim], axis=-1)
            dim += 1

        return kernel


class GaussianBlurring2D(tf.keras.layers.DepthwiseConv2D):
    """ Gaussian Blurring of 2D Map """
    def __init__(self, **kwargs):
        gaussian_blurr_initializer = GaussianBlurr2D(sigma=0.84)
        super(GaussianBlurring2D, self).__init__(
            **kwargs,
            strides=(1, 1),
            depth_multiplier=1,
            depthwise_initializer=gaussian_blurr_initializer,
            use_bias=False,
            trainable=False)
    
    def __call__(self, inputs):
        return super(GaussianBlurring2D, self).__call__(inputs)


class DropOutLayer(tf.keras.layers.Layer):
    """ DropOut Layer - Randomly drop and replace pixels
        Args:
            inputs: image from which to randomly drop and replace pixels
            backgrounds: image from which to take replacement pixels
    """
    def __init__(self, **kwargs):
        self.mult_layer = tf.keras.layers.Lambda(
            lambda x: tf.math.multiply(x[0], x[1]), trainable=False)

    def __call__(self, inputs, backgrounds, **kwargs):
        mask2d = tf.random.uniform(
            inputs.shape[1:3], minval=0, maxval=2, dtype=tf.int32)

        mask3d = tf.broadcast_to(
            tf.expand_dims(mask2d, -1), shape=inputs.shape[1:])
        mask3d = tf.cast(mask3d, tf.dtypes.float32)

        negative_mask3d = tf.math.abs(mask3d -1)

        masked_inputs = self.mult_layer([inputs, mask3d])
        masked_backgrounds = self.mult_layer([backgrounds, negative_mask3d])

        combined = tf.math.add(masked_inputs, masked_backgrounds)

        return combined
