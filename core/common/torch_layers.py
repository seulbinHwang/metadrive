from typing import Optional, Union, Dict
from diffusion_planner.utils.config import Config as DiffusionPlannerConfig

import gymnasium as gym
import torch
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from diffusion_planner.model.diffusion_planner import Diffusion_Planner_Encoder
from diffusion_planner.model.module.decoder import RouteEncoder


class DiffusionExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional network ("combined").

    :param observation_space:
    """

    def __init__(self,
                 observation_space: spaces.Dict,
                 features_dim: int = 1,
                 config: Optional[DiffusionPlannerConfig] = None) -> None:
        super().__init__(observation_space, features_dim)
        """
        Neighbors: MLP Mixer
        Lanes: MLP Mixer
        Static obj: MLP
        
        Navigation: MLP Mixer
        """
        self.encoder = Diffusion_Planner_Encoder(config)
        self.route_encoder = RouteEncoder(
            config.route_num,
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim)

    def forward(self, observations: TensorDict) -> Dict[str, torch.Tensor]:
        # TODO
        # add "lanes_speed_limit" (B, P, 1)
        # add "lanes_has_speed_limit" (B, P, 1)
        B, P, _, _ = observations['lanes'].shape
        observations['lanes_speed_limit'] = torch.ones(
            (B, P, 1), device=observations['lanes'].device)
        observations['lanes_speed_limit'] *= 100 / 3.6
        observations['lanes_has_speed_limit'] = torch.zeros(
            (B, P, 1), device=observations['lanes'].device)
        # encoder_outputs:
        # (B, self.token_dim(=agent_num + static_num + lane_num), self.hidden_dim)
        encoder_outputs = self.encoder(observations)

        route_lanes = observations['route_lanes']
        route_encoding = self.route_encoder(route_lanes)
        # TODO: dict 말고 torch.Tensor 로 출력해서, features_dim 을 의미있게 만들지 고민해보기
        outputs = {
            "encoding": encoder_outputs,
            "route_encoding": route_encoding
        }
        return outputs
