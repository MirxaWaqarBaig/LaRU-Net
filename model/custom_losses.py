# -*- coding: utf-8 -*-
"""
custom_losses.py
Contains custom loss functions including region-weighted loss, perceptual SSIM,
and edge preservation. Also initializes a partial VGG19 model for perceptual loss.
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

# Build a partial VGG19 model to extract perceptual features
# (block4_conv2 is an example layer; adapt if needed)
vgg = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer("block4_conv2").output)
vgg_model.trainable = False  # Freeze VGG19 layers

def region_weighted_loss(y_true, y_pred):
    """
    Applies higher weights for regions of interest (e.g., >0.5).
    """
    weight_map = tf.where(y_true > 0.5, 2.0, 1.0)
    return tf.reduce_mean(weight_map * tf.square(y_true - y_pred))

def perceptual_ssim_loss(y_true, y_pred):
    """
    Combines SSIM loss and VGG-based perceptual loss for high-quality image generation.
    """
    y_pred_resized = tf.image.resize(y_pred, tf.shape(y_true)[1:3])
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred_resized, max_val=1.0))
    
    # Convert grayscale to 3 channels for VGG
    y_true_rgb = tf.image.grayscale_to_rgb(y_true)
    y_pred_rgb = tf.image.grayscale_to_rgb(y_pred_resized)
    
    perceptual_loss = tf.reduce_mean(tf.square(vgg_model(y_true_rgb) - vgg_model(y_pred_rgb)))
    
    return 0.5 * ssim_loss + 0.5 * perceptual_loss

def edge_preservation_loss(y_true, y_pred):
    """
    Uses Sobel edges to encourage preserving edges in the output.
    """
    y_true_edges = tf.image.sobel_edges(y_true)
    y_pred_edges = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.square(y_true_edges - y_pred_edges))

def combined_loss(y_true, y_pred):
    """
    Weighted sum of the region-weighted, perceptual_ssim, and edge_preservation losses.
    """
    return (
        0.5  * region_weighted_loss(y_true, y_pred) +
        0.25 * perceptual_ssim_loss(y_true, y_pred) +
        0.25 * edge_preservation_loss(y_true, y_pred)
    )
