import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNN1DFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_input_channels=1, output_dim=128):
        # observation_space.shape = (N_past * state_dim,)
        self.n_past = 5  # Number of past observations stacked
        self.state_dim = observation_space.shape[0] // self.n_past
        super().__init__(observation_space, features_dim=output_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Determine output shape after CNN
        sample_input = torch.zeros((1, 1, self.n_past * self.state_dim))
        sample_out = self.cnn(sample_input.unsqueeze(1))
        self.linear = nn.Sequential(
            nn.Linear(sample_out.shape[1], output_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, N_past * state_dim)
        batch_size = observations.shape[0]
        obs = observations.view(batch_size, 1, -1)
        cnn_out = self.cnn(obs)
        return self.linear(cnn_out)
