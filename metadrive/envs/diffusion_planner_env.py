import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union, List
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
from metadrive.manager.record_manager import RecordManager
from metadrive.manager.replay_manager import ReplayManager
from metadrive.envs.metadrive_env import MetaDriveEnv, METADRIVE_DEFAULT_CONFIG
from metadrive.manager.traffic_manager import HistoricalBufferTrafficManager
from metadrive.manager.speed_limit_pg_map_manager import SpeedLimitPGMapManager
from metadrive.manager.object_manager import TrafficObjectManager
from metadrive.obs.diffusion_planner_obs import DiffusionPlannerObservation
from pathlib import Path

args_path = Path(
    '~/PycharmProjects/metadrive/checkpoints/args.json').expanduser()

from typing import Dict, TypeVar

K = TypeVar("K")  # 키 타입 (hashable)
V = TypeVar("V")  # 값 타입 (Any)


def merge_dicts(first: Dict[K, V], second: Dict[K, V]) -> Dict[K, V]:
    """
    두 딕셔너리를 병합하여 반환합니다.

    Args:
        first (Dict[K, V]): 우선순위가 낮은(기본) 딕셔너리.
        second (Dict[K, V]): 우선순위가 높은(덮어쓰기) 딕셔너리.

    Returns:
        Dict[K, V]: `first`와 `second`를 합친 새 딕셔너리.
            - 동일 키가 있을 경우 `second`의 값이 최종 결과에 반영됩니다.

    예시:
        >>> a = {"x": 1, "y": 2}
        >>> b = {"y": 99, "z": 3}
        >>> merge_dicts(a, b)
        {'x': 1, 'y': 99, 'z': 3}
    """
    return {**first, **second}


DIFFUSION_PLANNER_DEFAULT_CONFIG = dict(
    max_speed_kph_limit=120,
    min_speed_kph_limit=30,
    random_lane_width=True,
    random_lane_num=True,
    agent_observation=DiffusionPlannerObservation,
    accident_prob=1.0,
    traffic_mode=TrafficMode.Respawn,
    traffic_density=0.,
    random_spawn_lane_index=False,
    agent_policy=LQRPolicy,
    agent_configs={
        DEFAULT_AGENT:
            dict(
                use_special_color=True,
                spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 1),
            )
    },
    vehicle_config=dict(vehicle_model="bicycle_history_default",),
    traffic_vehicle_config=dict(
        max_acceleration_range=(2., 3.5),
        max_deceleration_range=(3., 9.),
    ))
diffusion_planner_config = DiffusionPlannerConfig(
    args_file=str(args_path)).to_dict()
DIFFUSION_PLANNER_DEFAULT_CONFIG = merge_dicts(DIFFUSION_PLANNER_DEFAULT_CONFIG,
                                               diffusion_planner_config)
# TODO: 위 코드 안먹음. 수정해야함


class DiffusionPlannerEnv(MetaDriveEnv):

    @classmethod
    def default_config(cls) -> Config:
        """
        BASE_DEFAULT_CONFIG + METADRIVE_DEFAULT_CONFIG + DIFFUSION_PLANNER_DEFAULT_CONFIG
        결과적으로는, 인자 config와 합쳐져서,
            BaseEnv.config 가 됨
            EngineCore.global_config 가 됨
        """
        config = super().default_config()
        config.update(DIFFUSION_PLANNER_DEFAULT_CONFIG)
        return config

    def setup_engine(self):
        """
        Engine setting after launching
        """
        self.engine.accept("r", self.reset)
        self.engine.accept("c", self.capture)
        self.engine.accept("p", self.stop)
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        self.engine.accept("]", self.next_seed_reset)
        self.engine.accept("[", self.last_seed_reset)
        self.engine.register_manager("agent_manager", self.agent_manager)
        self.engine.register_manager("record_manager", RecordManager())
        self.engine.register_manager("replay_manager", ReplayManager())
        self.engine.register_manager("map_manager", SpeedLimitPGMapManager())
        self.engine.register_manager("traffic_manager",
                                     HistoricalBufferTrafficManager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager",
                                         TrafficObjectManager())
