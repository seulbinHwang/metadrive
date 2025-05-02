import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Union
import colorsys
from core.diffusion_dppo.diffusion_ppo import DiffusionPPO
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.envs.diffusion_planner_env_w_npc import DiffusionPlannerEnv
from core.common.policies import DiffusionActorCriticPolicy
import time
import numpy as np
import math
# ── utils_vis.py ────────────────────────────────────────────
import numpy as np

RED = (1, 0, 0, 1)          # RGBA




def main():

    config = {
        "num_scenarios": 10,
        "start_seed": 105,
        "traffic_density": 0.1,
        "map": 5,
        "random_traffic": True,
        "use_render": True,  # <- 화면 창 활성화
        "debug": False,
        "accident_prob": 0.,
    }

    env = DiffusionPlannerEnv(config)
    model = DiffusionPPO(
        policy=DiffusionActorCriticPolicy,
        env=env,
        verbose=1,  # 학습 과정 콘솔 출력 (0:출력없음, 1:정보, 2:상세)
        tensorboard_log="./ppo_metadrive_tensorboard",  # TensorBoard 로그 디렉토리
    )

    N_EPISODES = 5
    obs, _ = env.reset()
    episode_num = 0
    while episode_num < N_EPISODES:
        """
        obs: (n, 19)
        """
        action, _ = model.predict(obs, deterministic=True)
        npc_predictions = model.get_npc_predictions(obs) # ( P-1, V_future = 80, 4)
        env.set_external_npc_actions(npc_predictions)
        """
        만약 VecEnv 였으면,
        env.env_method(method_name="set_external_npc_actions",
                           npc_actions=npc_predictions)
        """
        obs, reward, terminated, truncated, info = env.step(action)
        # done 이 된 환경만 기록-저장
        if terminated or truncated:
            episode_num += 1
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
