import json
import torch
from typing import Dict, Any
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.model.guidance.guidance_wrapper import GuidanceWrapper


class Config:

    def __init__(self, args_file):
        with open(args_file, 'r') as f:
            args_dict = json.load(f)

        for key, value in args_dict.items():
            setattr(self, key, value)
        self.state_normalizer = StateNormalizer(self.state_normalizer['mean'],
                                                self.state_normalizer['std'])
        self.observation_normalizer = ObservationNormalizer({
            k: {
                'mean': torch.as_tensor(v['mean']),
                'std': torch.as_tensor(v['std'])
            } for k, v in self.observation_normalizer.items()
        })
        self.guidance_fn = GuidanceWrapper()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert current Config object's public attributes into a dictionary.
        """
        result = {}
        # __dict__ 를 순회하면서 스페셜(Attribute)인 '__...' 들은 무시
        for attribute_name, attribute_value in self.__dict__.items():
            if attribute_name.startswith("__"):
                continue
            # 그 외 일반 값은 직접 저장
            result[attribute_name] = attribute_value

        return result
