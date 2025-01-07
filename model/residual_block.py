# -*- coding: utf-8 -*-
"""
residual_block.py
This file defines the residual block used in the encoder and decoder.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add

def residual_block(x, filters, kernel_size=3):
    """
    Standard residual block with two convolution layers and skip connection.
    Automatically adjusts the shortcut if filter dimensions differ.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters for convolution layers.
        kernel_size (int): Convolution kernel size.

    Returns:
        tf.Tensor: Output tensor after applying residual block.
    """
    shortcut = x
    
    # First Conv
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Second Conv
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    
    # Match channel dimensions
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding="same")(shortcut)
    
    # Add and ReLU
    x = Add()([x, shortcut])
    x = ReLU()(x)
    
    return x
