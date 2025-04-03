from stable_baselines3.ppo.policies import PPOPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from .custom_policy.cnn_feature_extractor import CNN1DFeatureExtractor

class CNNPPOPolicy(PPOPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=CNN1DFeatureExtractor,
                         features_extractor_kwargs=dict(n_input_channels=1, output_dim=128))
