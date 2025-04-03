import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN1DFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, stack_size=4, output_dim=128):
        super().__init__(observation_space, features_dim=output_dim)

        self.stack_size = stack_size
        self.state_dim = observation_space.shape[1]

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.state_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute size after CNN using a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros((1, self.stack_size, self.state_dim)).permute(0, 2, 1)  # shape: (1, state_dim, stack)
            dummy_out = self.cnn(dummy_input)
            n_flatten = dummy_out.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, output_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations: (batch, stack_size, state_dim) → CNN needs (batch, channels, time)
        x = observations.permute(0, 2, 1)
        return self.linear(self.cnn(x))
