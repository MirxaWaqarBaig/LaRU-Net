# -*- coding: utf-8 -*-
"""
gpu_setup.py
Handles GPU memory growth or other GPU configurations.
"""

import tensorflow as tf

def enable_gpu_memory_growth():
    """
    Enables memory growth for all available GPUs to prevent allocation errors.
    Call at the start of training or inference.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("GPU Setup Error:", e)
