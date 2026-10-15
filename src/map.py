"""Shared map utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from typing import Dict

class ModelMAP:

    def __init__(self, Visual_stimuli, Audio_stimuli, Spatial_disparity) -> None:
        self.visual_stimuli = Visual_stimuli
        self.audio_stimuli = Audio_stimuli
        self.spatial_disparity = Spatial_disparity
        self.disparity_threshold = 8
        mask = np.abs(self.visual_stimuli - self.audio_stimuli) < self.disparity_threshold
        self.visual_stimuli_unity = self.visual_stimuli[mask]
        self.audio_stimuli_unity = self.audio_stimuli[mask]
        self.visual_stimuli_nonunity = self.visual_stimuli[~mask]
        self.audio_stimuli_nonunity = self.audio_stimuli[~mask]

    def model_unity(self):
        w1_unity = np.var(self.visual_stimuli_unity) / (np.var(self.audio_stimuli_unity) + np.var(self.visual_stimuli_unity))
        w2_unity = np.var(self.audio_stimuli_unity) / (np.var(self.audio_stimuli_unity) + np.var(self.visual_stimuli_unity))
        return len(self.audio_stimuli_unity) / (len(self.audio_stimuli_unity) + len(self.audio_stimuli_nonunity)) * (w1_unity * self.audio_stimuli_unity + w2_unity * self.visual_stimuli_unity) + (1 - len(self.audio_stimuli_unity) / (len(self.audio_stimuli_unity) + len(self.audio_stimuli_nonunity))) * self.audio_stimuli_unity

    def model_nonunity(self):
        return self.audio_stimuli_nonunity
