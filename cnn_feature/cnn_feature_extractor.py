import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CNN1DFeatureExtractor(BaseFeaturesExtractor):
    """
    A 1D CNN feature extractor for stacked time-series observations.
    Converts (stack_size, state_dim) into a flattened feature vector for the policy/value network.
    """

    def __init__(self, observation_space: gym.spaces.Box, output_dim: int = 128):
        # Extract shape
        stack_size, state_dim = observation_space.shape
        self.stack_size = stack_size
        self.state_dim = state_dim

        # Temporary CNN to infer output size
        cnn = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Infer flattened CNN output size using dummy input
        with torch.no_grad():
            sample_input = torch.zeros((1, stack_size, state_dim)).permute(0, 2, 1)  # (batch, state_dim, stack_size)
            flattened_dim = cnn(sample_input).shape[1]

        super().__init__(observation_space, features_dim=output_dim)

        # Save CNN and linear projection
        self.cnn = cnn
        self.linear = nn.Sequential(
            nn.Linear(flattened_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, stack_size, state_dim)
        x = observations.permute(0, 2, 1)  # → (batch_size, state_dim, stack_size)
        return self.linear(self.cnn(x))
