# -*- coding: utf-8 -*-
"""
plotting.py
Includes functions to plot and save training history (loss, MAE, etc.).
"""

import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_dir=None):
    """
    Plots the training history (Loss and MAE) and saves the figures if save_dir is specified.

    Args:
        history (History): History object from model.fit()
        save_dir (str): Directory to save plots. If None, plots are not saved.
    """
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot Loss
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss across Epochs')
    plt.legend()

    if save_dir:
        loss_path = os.path.join(save_dir, 'training_loss.png')
        plt.savefig(loss_path, dpi=600, format='png')
    plt.close()

    # Plot MAE
    plt.figure()
    plt.plot(epochs, history.history['mae'], 'g', label='Training MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training MAE across Epochs')
    plt.legend()

    if save_dir:
        mae_path = os.path.join(save_dir, 'training_mae.png')
        plt.savefig(mae_path, dpi=600, format='png')
    plt.close()

def plot_combined_training_history(history, save_dir=None):
    """
    Plots the training Loss and MAE on the same figure, optionally saves as PNG.

    Args:
        history (History): History object from model.fit()
        save_dir (str): Directory to save combined plot. If None, not saved.
    """
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"  # Example font setting

    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['mae'], 'g', label='Training MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Loss and MAE across Epochs')
    plt.legend()
    
    if save_dir:
        combined_path = os.path.join(save_dir, 'combined_training_history.png')
        plt.savefig(combined_path, dpi=1200, format='png')
    plt.show()
