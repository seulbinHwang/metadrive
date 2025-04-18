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


def visualize_entire_road_network(road_network: NodeRoadNetwork,
                                  ego_x: float,
                                  ego_y: float,
                                  ego_yaw: float,
                                  sampling_interval: float = 1.0,
                                  roi_length: float = 200.0,
                                  ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal", "box")

    all_lanes = road_network.get_all_lanes()
    for lane in all_lanes:
        ln_len = lane.length
        if ln_len < 1e-6:
            continue
        w = getattr(lane, "width", 3.5)
        half_w = w * 0.5

        s_vals = np.arange(0.0, ln_len + 1e-9, sampling_interval)
        if len(s_vals) < 2:
            s_vals = np.array([0.0, ln_len])

        N = len(s_vals)
        center_w = np.zeros((N, 2), dtype=np.float32)
        left_w = np.zeros((N, 2), dtype=np.float32)
        right_w = np.zeros((N, 2), dtype=np.float32)

        for i, s_val in enumerate(s_vals):
            cx, cy = lane.position(s_val, 0.0)
            lx, ly = lane.position(s_val, +half_w)
            rx, ry = lane.position(s_val, -half_w)
            center_w[i] = (cx, cy)
            left_w[i] = (lx, ly)
            right_w[i] = (rx, ry)

        cx_l, cy_l = _to_local_coords_batch(center_w[:, 0], center_w[:, 1], ego_x, ego_y, ego_yaw)
        lx_l, ly_l = _to_local_coords_batch(left_w[:, 0], left_w[:, 1], ego_x, ego_y, ego_yaw)
        rx_l, ry_l = _to_local_coords_batch(right_w[:, 0], right_w[:, 1], ego_x, ego_y, ego_yaw)

        ax.plot(cx_l, cy_l, "--", color="white", linewidth=1.0)
        ax.plot(lx_l, ly_l, "-", color="white", linewidth=1.2)
        ax.plot(rx_l, ry_l, "-", color="white", linewidth=1.2)

    ax.set_facecolor("black")
    c_len = roi_length / 20.0
    c_wid = roi_length / 100.0
    rx = -c_len * 0.5
    ry = -c_wid * 0.5
    rect = Rectangle((rx, ry), c_len, c_wid, edgecolor="white", facecolor="none", zorder=5)
    ax.add_patch(rect)
    ax.arrow(0, 0, c_len, 0, width=0.002 * roi_length, head_width=0.01 * roi_length,
             color="white", length_includes_head=True, zorder=6)


def visualize_lanes_array(lanes_array: np.ndarray,
                          length: float,
                          color="orange",
                          ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

    max_lane_num = lanes_array.shape[0]
    for lane_idx in range(max_lane_num):
        lane_data = lanes_array[lane_idx]
        center_xy = lane_data[:, 0:2]
        left_off = lane_data[:, 4:6]
        right_off = lane_data[:, 6:8]

        left_xy = center_xy + left_off
        right_xy = center_xy + right_off

        ax.plot(center_xy[:, 0], center_xy[:, 1], "--", color=color)
        ax.plot(left_xy[:, 0], left_xy[:, 1], "-", color=color)
        ax.plot(right_xy[:, 0], right_xy[:, 1], "-", color=color)

    # 정사각형 테두리
    square = Rectangle(
        (-length//2, -length//2),
        length,
        length,
        linewidth=2,
        edgecolor='cyan',
        facecolor='none')
    ax.add_patch(square)

def visualize_static_objects(ax,
                             static_objects: np.ndarray,
                             edgecolor="purple",
                             fill=False):
    N = static_objects.shape[0]
    for i in range(N):
        row = static_objects[i]
        x, y     = row[0], row[1]
        cos_h    = row[2]
        sin_h    = row[3]
        w        = row[4]
        l        = row[5]
        heading  = math.atan2(sin_h, cos_h)
        heading_deg = math.degrees(heading)

        anchor_x = x - l/2.0
        anchor_y = y - w/2.0

        rect = Rectangle((anchor_x, anchor_y),
                         l,
                         w,
                         angle=heading_deg,
                         edgecolor=edgecolor,
                         facecolor=edgecolor if fill else "none",
                         lw=1.5,
                         zorder=10,
                         alpha=1.0)
        ax.add_patch(rect)

def visualize_neighbors_history(ax, neighbors_history: np.ndarray):
    """
    neighbors_history: (max_num, max_time_steps, 11)
    [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, 1, 0, 0]
    """
    max_num = neighbors_history.shape[0]
    max_time_steps = neighbors_history.shape[1]

    # 색상 할당 (단순 HSV 변환)
    import colorsys
    colors = []
    for i in range(max_num):
        hue = (i * 0.15) % 1.0
        color_rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(color_rgb)

    for i in range(max_num):
        color = colors[i]
        for t in range(max_time_steps):
            row = neighbors_history[i, t]
            x = row[0]
            y = row[1]
            cos_h = row[2]
            sin_h = row[3]
            w = row[6]
            l = row[7]
            if (abs(x)+abs(y)+abs(cos_h)+abs(sin_h)+abs(w)+abs(l))<1e-8:
                continue
            heading = math.atan2(sin_h, cos_h)
            heading_deg = math.degrees(heading)
            anchor_x = x - l/2.0
            anchor_y = y - w/2.0

            rect = Rectangle((anchor_x, anchor_y),
                             l,
                             w,
                             angle=heading_deg,
                             edgecolor=color,
                             facecolor="none",
                             lw=1.0,
                             zorder=12,
                             alpha=0.8)
            ax.add_patch(rect)

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
    # 2) env.step 반복
    action = [0.0, 0.3]
    for i in range(120):
        env.step(action)
    observations = env.observations
    observation = observations["default_agent"]
    ego = env.vehicle

    # 1) 특정 ref_lane(현재 + 다음)에서 추출한 path (로컬 좌표계)
    ref_path = get_ref_path(ego)



    # 3) 시각화
    roadnet = env.current_map.road_network

    obs_dict = observation.observe(ego)
    lanes_array = obs_dict["lanes_array"]           # (70, 20, 12)
    nav_lanes_array = obs_dict["nav_lanes_array"]   # (25, 20, 12)
    static_objects = obs_dict["static_objects"]      # (N, 10)
    neighbors_history = obs_dict["neighbors_history"]# (32, 21, 11)
    roi_len = observation.lane_roi_length

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # (a) RoadNetwork
    visualize_entire_road_network(roadnet,
                                  ego_x, ego_y, ego_yaw,
                                  sampling_interval=1.0,
                                  roi_length=roi_len,
                                  ax=ax)

    # (b) lane_array
    visualize_lanes_array(lanes_array, length=roi_len, color="orange", ax=ax)
    visualize_lanes_array(nav_lanes_array, length=roi_len, color="red", ax=ax)

    # (c) static
    visualize_static_objects(ax, static_objects, edgecolor="purple", fill=False)

    # (d) neighbor history
    visualize_neighbors_history(ax, neighbors_history)

    # (e) ref_path 시각화 (초록선 + 방향 화살표)
    if ref_path.shape[0]>0:
        # ref_path: (N,4) => [x_local,y_local, cosθ, sinθ]
        ax.plot(ref_path[:,0], ref_path[:,1],
                color="white", linewidth=1.0,
                label="ref_path")

        # 화살표(방향) 표시: 작은 scaled arrow로 표현
        arrow_scale = 0.3  # 화살표 길이
        for i in range(0, ref_path.shape[0], 5):
            px, py, cth, sth = ref_path[i]
            ax.arrow(px, py,
                     cth*arrow_scale,
                     sth*arrow_scale,
                     width=1.0,
                     head_width=1.0,
                     color="white",
                     length_includes_head=True,
                     zorder=9,
                     alpha=0.9)
    # 스타일 정리
    ax.set_facecolor("black")
    ax.set_title("ROI Lanes + Static Objects + NeighborsHistory + ref_path")
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.legend(loc="upper right", facecolor="black", edgecolor="white", labelcolor="white")
    plt.savefig("lanes_static_neigh_history_refpath.png", dpi=150, facecolor="black")
    plt.close()
    print("Saved lanes_static_neigh_history_refpath.png")

    env.close()


if __name__ == "__main__":
    main()
