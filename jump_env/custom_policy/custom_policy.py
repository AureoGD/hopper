from stable_baselines3.common.policies import ActorCriticPolicy
from jump_env.custom_policy.cnn_feature_extractor import CNN1DFeatureExtractor


class CNNPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", CNN1DFeatureExtractor)
        kwargs.setdefault("features_extractor_kwargs", dict(stack_size=10, output_dim=128))
        super().__init__(*args, **kwargs)
