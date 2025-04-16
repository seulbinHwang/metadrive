import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union
from metadrive.policy.lqr_policy import LQRPolicy
import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config
from diffusion_planner.utils.config import Config as DiffusionPlannerConfig

from metadrive.envs.metadrive_env import MetaDriveEnv

from metadrive.obs.diffusion_planner_obs import DiffusionPlannerObservation


def merge_dictionaries(*dicts: dict) -> dict:
    """
    여러 개의 dictionary를 하나로 합쳐 반환합니다.

    Args:
        dicts (dict): 병합할 딕셔너리들의 가변 인자.

    Returns:
        dict: 모든 딕셔너리를 병합한 결과로 생성된 딕셔너리.

    Raises:
        KeyError: 두 개 이상의 딕셔너리에서 같은 키가 발견될 경우 발생.
    """
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                raise KeyError(f"키 '{key}' 가 중복되었습니다.")
            merged_dict[key] = value

    return merged_dict


DIFFUSION_PLANNER_DEFAULT_CONFIG = dict(
    agent_observation=DiffusionPlannerObservation, accident_prob=1.0,
traffic_mode=TrafficMode.Respawn,
    traffic_density=0.5,
agent_policy=LQRPolicy,
vehicle_config=dict(
vehicle_model="bicycle_default",)
)

diffusion_planner_config = DiffusionPlannerConfig(
    args_file="/home/user/PycharmProjects/metadrive/checkpoints/args.json"
).to_dict()
DIFFUSION_PLANNER_DEFAULT_CONFIG = merge_dictionaries(
    DIFFUSION_PLANNER_DEFAULT_CONFIG, diffusion_planner_config)
# TODO: 위 코드 안먹음. 수정해야함


class DiffusionPlannerEnv(MetaDriveEnv):

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(DIFFUSION_PLANNER_DEFAULT_CONFIG)
        return config

    def __init__(self, config: Union[dict, None] = None):
        super().__init__(config)
