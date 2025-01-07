# -*- coding: utf-8 -*-
"""
data_loader.py
Handles loading and preprocessing of images (training dataset, etc.).
"""

import os
import cv2
import numpy as np

IMG_HEIGHT = 256
IMG_WIDTH = 256

def load_and_preprocess_image(image_path):
    """
    Loads a grayscale image from disk, resizes it, normalizes to [0,1],
    and reshapes to (H, W, 1).
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: A (H, W, 1) float32 tensor with values in [0,1].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Channel dimension
    return img

def load_dataset(normal_dir, enhanced_dir):
    """
    Loads paired normal and enhanced images from corresponding directories.
    Assumes filenames are identical in both folders.

    Args:
        normal_dir (str): Directory with normal images.
        enhanced_dir (str): Directory with enhanced images.

    Returns:
        tuple: (normal_images, enhanced_images) as numpy arrays.
    """
    normal_images = []
    enhanced_images = []
    
    for file_name in os.listdir(normal_dir):
        normal_path = os.path.join(normal_dir, file_name)
        enhanced_path = os.path.join(enhanced_dir, file_name)
        
        if os.path.exists(enhanced_path):
            normal_img = load_and_preprocess_image(normal_path)
            enhanced_img = load_and_preprocess_image(enhanced_path)
            normal_images.append(normal_img)
            enhanced_images.append(enhanced_img)
    
    return np.array(normal_images), np.array(enhanced_images)
