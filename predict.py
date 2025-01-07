# -*- coding: utf-8 -*-
"""
predict.py
Script to run inference on a single image (or multiple images) using the trained model.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from model.custom_losses import (
    combined_loss, 
    region_weighted_loss, 
    perceptual_ssim_loss, 
    edge_preservation_loss
)
from utils.data_loader import IMG_HEIGHT, IMG_WIDTH

def load_and_preprocess_test_image(image_path):
    """
    Loads a single test image in grayscale, resizes it, and 
    prepares it for model inference.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    # Get the directory containing this script
    script_dir = os.path.dirname(script_path)

    # ----------------------------
    # Set relative paths
    # ----------------------------
    # Folder with trained model
    trained_model_dir = os.path.join(script_dir)
    model_name = 'Residual_UNet_Model.h5'
    model_path = os.path.join(trained_model_dir, model_name)

    # Folder with test images
    test_images_dir = os.path.join(script_dir, 'test_images')
    test_image_name = '1 (10).png'  # Example test image
    test_image_path = os.path.join(test_images_dir, test_image_name)

    # ----------------------------
    # Verify paths
    # ----------------------------
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.isfile(test_image_path):
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    # ----------------------------
    # Load model
    # ----------------------------
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "combined_loss": combined_loss,
            "region_weighted_loss": region_weighted_loss,
            "perceptual_ssim_loss": perceptual_ssim_loss,
            "edge_preservation_loss": edge_preservation_loss,
        },
        compile=False
    )
    print("Model loaded successfully.")

    # ----------------------------
    # Load and predict
    # ----------------------------
    test_image = load_and_preprocess_test_image(test_image_path)
    enhanced_image = model.predict(test_image)

    # Convert images back to display range [0, 255]
    original_image = (test_image[0, :, :, 0] * 255.0).astype(np.uint8)
    enhanced_image = (enhanced_image[0, :, :, 0] * 255.0).astype(np.uint8)

    # ----------------------------
    # Visualization
    # ----------------------------
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Test Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title("Enhanced Test Image")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
