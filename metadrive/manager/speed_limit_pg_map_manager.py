# speed_limit_pg_map_manager.py
"""
Speed-Limit aware PGMapManager
==============================
`SpeedLimitPGMapManager`는 기존 `PGMapManager` 기능을 그대로 쓰면서
에피소드(=reset) 마다 모든 Lane 의 `speed_limit`(km/h 단위)을
지정한 범위에서 무작위로 재설정해 준다.

● 켜고 끄기
    engine.global_config["randomize_speed_limit"] = True / False

● 범위 지정 (km/h)
    engine.global_config["speed_limit_kmh_range"] = (40, 120)

설정을 지정하지 않으면 기본값이 이용된다.
"""

from typing import Tuple
import numpy as np

from metadrive.manager.pg_map_manager import PGMapManager


class SpeedLimitPGMapManager(PGMapManager):
    """
    PGMapManager + Lane speed-limit randomizer
    """

    def __init__(self):
        super().__init__()
        self.speed_limit_kph = None

    def add_random_to_map(self, map_config):
        map_config = super().add_random_to_map(map_config)
        self.speed_limit_kph = self.np_random.rand() * (
            self.engine.global_config["max_speed_kph_limit"] -
            self.engine.global_config["min_speed_kph_limit"]) + \
            self.engine.global_config["min_speed_kph_limit"]
        return map_config

    # --------------------------------------------------------------------- #
    #                       RESET:  맵 준비 + 속도 랜덤화                    #
    # --------------------------------------------------------------------- #
    def reset(self):
        """
        1) (super) : 시드에 맞는 맵을 로드/생성하여 `self.current_map` 세팅
        2) 필요하면 모든 Lane 의 `speed_limit` 값을 랜덤으로 갱신
        """
        super().reset()  # → self.current_map 이 준비되어 있음
        self._randomize_speed_limits()

    # --------------------------------------------------------------------- #
    #                          내부: speed-limit 랜덤화                      #
    # --------------------------------------------------------------------- #
    def _randomize_speed_limits(self) -> None:
        """
        모든 Lane 의 `speed_limit` 을 [min, max] km/h 범위에서 무작위 설정
        (시드: self.np_random – 에피소드 재현성 보장)
        """
        if self.current_map is None or self.speed_limit_kph is None:
            return

        for lane in self.current_map.road_network.get_all_lanes():
            # 필요한 경우: 커브·교차로 등 타입별로 조정 가능
            lane.speed_limit = float(self.speed_limit_kph)
