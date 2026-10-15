"""Shared training utilities."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from typing import Dict

class Trainer:

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int=1000, log_interval: int=100):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.log_interval = log_interval
        self.loss_history = []
        hidden_shape = model.hidden.weight.shape
        output_shape = model.output.weight.shape
        self.weights_history = {'hidden': np.zeros((hidden_shape[0], hidden_shape[1], epochs)), 'output': np.zeros((output_shape[0], output_shape[1], epochs))}
        self.device = next(model.parameters()).device
        self.model.to(self.device)

    def train(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Train the model and track loss and weight evolution."""
        X, Y = (X.to(self.device), Y.to(self.device))
        self.model.train()
        for epoch in range(self.epochs):
            outputs = self.model(X)
            loss = self.criterion(outputs.squeeze(), Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            self.update_weights(epoch)
            if (epoch + 1) % self.log_interval == 0 or epoch + 1 == self.epochs:
                print(f'Epoch {epoch + 1}/{self.epochs} - Loss: {loss.item():.6f}')

    def update_weights(self, epoch: int) -> None:
        """Extract and store weights in pre-allocated arrays at the given epoch index."""
        with torch.no_grad():
            self.weights_history['hidden'][:, :, epoch] = self.model.hidden.weight.detach().cpu().numpy()
            self.weights_history['output'][:, :, epoch] = self.model.output.weight.detach().cpu().numpy()

    def get_loss_history(self) -> np.ndarray:
        """Return the loss history over epochs as a NumPy array."""
        return np.array(self.loss_history)

    def get_weights_history(self) -> Dict[str, np.ndarray]:
        """Return the structured weight evolution over epochs."""
        return self.weights_history
