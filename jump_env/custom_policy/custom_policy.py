from stable_baselines3.common.policies import ActorCriticPolicy
from custom_policy.cnn_feature_extractor import CNN1DFeatureExtractor

class CNNPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CNN1DFeatureExtractor,
            features_extractor_kwargs=dict(stack_size=4, output_dim=128)
        )
