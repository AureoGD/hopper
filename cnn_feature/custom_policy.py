# cnn_feature/custom_policy.py

from stable_baselines3.common.policies import ActorCriticPolicy
from cnn_feature.cnn_feature_extractor import CNN1DFeatureExtractor


class CNNPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        cnn_kwargs = dict(output_dim=128)  # ← change this if needed
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CNN1DFeatureExtractor,
            features_extractor_kwargs=cnn_kwargs,
        )
