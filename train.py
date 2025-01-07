# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
train.py
Script to train the Laplacian Attention U-Net model.
"""

import os
import tensorflow as tf
from model.LaR_Unet_architecture import build_laplacian_attention_unet
from model.custom_losses import combined_loss, region_weighted_loss, perceptual_ssim_loss, edge_preservation_loss
from utils.data_loader import load_dataset
from utils.plotting import plot_training_history, plot_combined_training_history
from utils.gpu_setup import enable_gpu_memory_growth

# -------------------------
# Determine Script Directory
# -------------------------
# Get the absolute path of the current script
script_path = os.path.abspath(__file__)
# Get the directory containing this script
script_dir = os.path.dirname(script_path)

# -------------------------
# Configuration using Relative Paths
# -------------------------
normal_image_dir = os.path.join(script_dir, 'All_PNG')
enhanced_image_dir = os.path.join(script_dir, 'All_Enhanced')
save_model_dir = os.path.join(script_dir, 'Saved_Model')

# Ensure that the save_model_dir exists; if not, create it
os.makedirs(save_model_dir, exist_ok=True)

IMG_HEIGHT = 256
IMG_WIDTH = 256

EPOCHS = 50
BATCH_SIZE = 2
#%%
# -------------------------
# GPU Setup
# -------------------------
enable_gpu_memory_growth()

# -------------------------
# Load Dataset
# -------------------------
normal_images, enhanced_images = load_dataset(normal_image_dir, enhanced_image_dir)

# -------------------------
# Build Model
# -------------------------
model = build_laplacian_attention_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
model.compile(optimizer='adam', loss=combined_loss, metrics=['mae'])
model.summary()
#%%
# -------------------------
# Train the Model
# -------------------------
history = model.fit(
    x=normal_images,
    y=enhanced_images,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# -------------------------
# Save the Model
# -------------------------
model_name = 'Residual_UNet_Model.h5'
model_path = os.path.join(save_model_dir, model_name)
model.save(model_path)
print(f"Model saved to: {model_path}")

# -------------------------
# Plot Training History
# -------------------------
plot_training_history(history, save_dir=save_model_dir)
plot_combined_training_history(history, save_dir=save_model_dir)
