# -*- coding: utf-8 -*-
"""
LaR_Unet_architecture.py
Defines the full LaRU-Net architecture with Laplacian Detail Enhancement Block
and attention-augmented skip connections.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, MaxPooling2D, Conv2DTranspose, Concatenate, Multiply, Conv2D
)
from tensorflow.keras.models import Model

# Import local modules
from .residual_block import residual_block
from .detail_enhancement_block import detail_enhancement_block

def attention_skip_connection(encoder_output, decoder_input, filters):
    """
    Applies a simple attention mechanism on the encoder output and concatenates
    with the decoder input.

    Args:
        encoder_output (tf.Tensor): Output feature map from encoder.
        decoder_input (tf.Tensor): Input feature map in decoder.
        filters (int): Number of filters for the attention map.

    Returns:
        tf.Tensor: Concatenated result of attention-weighted encoder output and decoder input.
    """
    attention = Conv2D(filters, (1, 1), activation="sigmoid")(encoder_output)
    attended_output = Multiply()([encoder_output, attention])
    return Concatenate()([decoder_input, attended_output])

def build_laplacian_attention_unet(input_shape=(256, 256, 1)):
    """
    Builds a U-Net with Laplacian-based Detail Enhancement Block in the bottleneck
    and attention-augmented skip connections.

    Args:
        input_shape (tuple): Shape of the input (H, W, C). Default is (256, 256, 1).

    Returns:
        tf.keras.Model: Compiled U-Net model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = residual_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = residual_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = residual_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = residual_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck (Detail Enhancement Block)
    b = detail_enhancement_block(p4, 1024)

    # Decoder
    u1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(b)
    u1 = attention_skip_connection(c4, u1, 512)
    u1 = residual_block(u1, 512)

    u2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(u1)
    u2 = attention_skip_connection(c3, u2, 256)
    u2 = residual_block(u2, 256)

    u3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(u2)
    u3 = attention_skip_connection(c2, u3, 128)
    u3 = residual_block(u3, 128)

    u4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(u3)
    u4 = attention_skip_connection(c1, u4, 64)
    u4 = residual_block(u4, 64)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u4)

    model = Model(inputs, outputs, name="LaplacianAttention_UNet")

    return model
