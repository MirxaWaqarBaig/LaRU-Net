# -*- coding: utf-8 -*-
"""
detail_enhancement_block.py
This file defines the Laplacian-based Detail Enhancement Block,
including attention mechanism and residual connections.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Multiply, Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MinMaxNorm

def detail_enhancement_block(x, filters):
    """
    Applies a Laplacian filter (initialized with custom kernel),
    includes an attention mechanism, and returns the enhanced features.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of output filters.

    Returns:
        tf.Tensor: Enhanced feature map after Laplacian filter and attention.
    """
    # Laplacian kernel for initialization
    laplacian_kernel = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=np.float32).reshape(3, 3, 1, 1)
    
    input_channels = x.shape[-1]  # Number of input channels
    laplacian_kernel = np.repeat(laplacian_kernel, input_channels, axis=2)  # Match input channels
    laplacian_kernel = np.repeat(laplacian_kernel, filters, axis=3)        # Match output channels
    
    kernel_initializer = tf.keras.initializers.Constant(laplacian_kernel)
    
    # Apply Laplacian filter to emphasize details
    laplacian = Conv2D(
        filters, (3, 3), padding="same",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=l2(1e-4),
        kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0),
        name="laplacian_conv"
    )(x)
    laplacian = BatchNormalization()(laplacian)
    laplacian = ReLU()(laplacian)
    
    # Enhanced attention mechanism
    attention = Conv2D(filters, (1, 1), activation="sigmoid")(laplacian)
    enhanced_features = Multiply()([laplacian, attention])
    
    # Match dimensions of x if necessary
    if x.shape[-1] != filters:
        x = Conv2D(filters, (1, 1), padding="same")(x)
    
    # Residual connection
    residual = Add()([x, enhanced_features])
    scaled_residual = ReLU()(residual)
    
    return scaled_residual
