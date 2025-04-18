import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Union
import colorsys

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.envs.diffusion_planner_env import DiffusionPlannerEnv

import numpy as np
import math


def extract_centerline_in_ego_frame(
        lane,
        ego_position_world: np.ndarray,
        ego_heading_world: float,
        step: float = 0.5,
        M_max: int = 200
) -> np.ndarray:
    """
    lane: StraightLane 혹은 CircularLane 객체 (AbstractLane 상속)
    ego_position_world: shape=(2,)  - (전역 좌표계에서) ego 차량의 (x, y) 위치
    ego_heading_world: float       - (전역 좌표계에서) ego 차량의 heading (rad)
    step: float = 0.5             - 각 점 사이의 간격(m)
    M_max: int = 200              - 최대 몇 개의 점을 저장할지 제한

    return: shape=(M,4) numpy array
      - M은 추출된 점의 개수 (최대 M_max)
      - 각 row = [x_local, y_local, cos_heading_local, sin_heading_local]
        (Ego 좌표계에서의 값)
    """
    s_ego, r_ego = lane.local_coordinates(ego_position_world)
    s_current = max(s_ego, 0.0)
    s_list = []
    while s_current <= lane.length and len(s_list) < M_max:
        s_list.append(s_current)
        s_current += step
    if not s_list:
        return np.zeros((0, 4), dtype=np.float32)

    out_list = []
    cos_ego = math.cos(ego_heading_world)
    sin_ego = math.sin(ego_heading_world)

    for s in s_list:
        world_xy = lane.position(s, 0)
        world_yaw = lane.heading_theta_at(s)

        dx = world_xy[0] - ego_position_world[0]
        dy = world_xy[1] - ego_position_world[1]
        x_local = dx * cos_ego + dy * sin_ego
        y_local = -dx * sin_ego + dy * cos_ego

        yaw_local = world_yaw - ego_heading_world
        yaw_local = (yaw_local + math.pi) % (2 * math.pi) - math.pi

        c_h = math.cos(yaw_local)
        s_h = math.sin(yaw_local)

        out_list.append((x_local, y_local, c_h, s_h))

    return np.array(out_list, dtype=np.float32)


def _to_local_coords_batch(px: np.ndarray, py: np.ndarray, ego_x: float,
                           ego_y: float, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    dx = px - ego_x
    dy = py - ego_y
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return local_x, local_y


def _rotate_vectors_batch(vx: np.ndarray, vy: np.ndarray, ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    x2 = vx * c + vy * s
    y2 = -vx * s + vy * c
    return x2, y2


def get_ref_path(ego: BaseVehicle) -> np.ndarray: # (80, 4)

    current_lane = ego.navigation.current_lane
    current_ref_lanes: list = ego.navigation.current_ref_lanes
    current_lane_idx = -1
    current_ref_lane = None
    for i, lane in enumerate(current_ref_lanes):
        if lane == current_lane:
            current_lane_idx = i
            current_ref_lane = lane
            break
    next_ref_lanes = ego.navigation.next_ref_lanes
    next_ref_lane = next_ref_lanes[current_lane_idx]

    # Ego 자세
    ego_x, ego_y = ego.position
    ego_yaw = ego.heading_theta

    # 첫 번째 레인에서 path 일부
    ref_path1 = extract_centerline_in_ego_frame(current_ref_lane,
                                                np.array([ego_x, ego_y], dtype=float),
                                                ego_yaw,
                                                step=0.5,
                                                M_max=80)
    ref_path_number = ref_path1.shape[0]
    leftover_number_from_max = 80 - ref_path_number

    # 두 번째 레인
    ref_path2 = np.zeros((0,4), dtype=np.float32)
    if leftover_number_from_max>0:
        ref_path2 = extract_centerline_in_ego_frame(next_ref_lane,
                                                    np.array([ego_x, ego_y], dtype=float),
                                                    ego_yaw,
                                                    step=0.5,
                                                    M_max=leftover_number_from_max)
    # 이어붙임
    ref_path = np.concatenate((ref_path1, ref_path2), axis=0)

    # 남은 slot이 있으면 마지막 점 복사
    leftover_number_from_max = 80 - ref_path.shape[0]
    if leftover_number_from_max>0 and ref_path.shape[0]>0:
        last_point = ref_path[-1]
        tile_pts = np.tile(last_point, (leftover_number_from_max,1))
        ref_path = np.concatenate((ref_path, tile_pts), axis=0)
    return ref_path
def main():
    from metadrive.envs.diffusion_planner_env import DiffusionPlannerEnv
    config = {
        "num_scenarios": 1,
        "start_seed": 0,
        "map": 5,
        "random_traffic": True,
        "use_render": False,
    }

    env = DiffusionPlannerEnv(config)
    env.reset()
    ego = env.vehicle
    ref_path = get_ref_path(ego)

    # 2) env.step 반복
    for i in range(120):
        env.step(ref_path)


if __name__ == "__main__":
    main()
