from .common import save_figure
"""Shared plotting utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from typing import Dict

def plot_weight_evolution(weights_history: dict) -> None:
    """Plot the weight evolution for each neuron in the hidden and output layers.
    
    Args:
        weights_history (dict): Dictionary containing weight evolution history.
            - weights_history["hidden"] -> Shape: (hidden_nodes, input_features, epochs)
            - weights_history["output"] -> Shape: (output_nodes, hidden_features, epochs)
    """
    epochs = weights_history['hidden'].shape[2]
    x_axis = np.arange(epochs)
    hidden_nodes, input_features, _ = weights_history['hidden'].shape
    fig, axes = plt.subplots(hidden_nodes, 1, figsize=(10, hidden_nodes * 3))
    fig.suptitle('Hidden Layer Weights Evolution', fontsize=14)
    if hidden_nodes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for j in range(input_features):
            if j == 0:
                lab = 'Visual Stimuli'
                col = 'orange'
            elif j == 1:
                lab = 'Audio Stimuli'
                col = 'blue'
            else:
                raise Exception('THE NUMBER OF STIMULI IS 2 OR LESS!')
            ax.plot(x_axis, weights_history['hidden'][i, j, :], label=lab, color=col, linewidth=3)
        ax.set_title(f'Hidden Neuron {i}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Weight')
        ax.set_ylim(-2, 2)
        ax.legend()
        ax.grid()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure()
    output_nodes, hidden_features, _ = weights_history['output'].shape
    fig, axes = plt.subplots(output_nodes, 1, figsize=(10, output_nodes * 3))
    fig.suptitle('Output Layer Weights Evolution', fontsize=14)
    if output_nodes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for j in range(hidden_features):
            ax.plot(x_axis, weights_history['output'][i, j, :], label=f'Hidden {j}', linewidth=3)
        ax.set_title(f'Output Neuron {i}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Weight')
        ax.set_ylim(-2, 5)
        ax.grid()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure()
