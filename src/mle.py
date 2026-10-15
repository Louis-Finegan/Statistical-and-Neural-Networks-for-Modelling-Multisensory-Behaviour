"""Shared mle utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from typing import Dict

class ModelMLE:

    def __init__(self, audio_stimuli, visual_stimuli) -> None:
        self.audio_stimuli = audio_stimuli
        self.visual_stimuli = visual_stimuli

    def audio_weight(self):
        return np.var(self.visual_stimuli) / (np.var(self.audio_stimuli) + np.var(self.visual_stimuli))

    def visual_weight(self):
        return np.var(self.audio_stimuli) / (np.var(self.audio_stimuli) + np.var(self.visual_stimuli))

    def model(self):
        return self.audio_weight() * self.audio_stimuli + self.visual_weight() * self.visual_stimuli
