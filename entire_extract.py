import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Union
import colorsys

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.envs.diffusion_planner_env import DiffusionPlannerEnv


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
    rect = Rectangle((rx, ry), c_len, c_wid, edgecolor="lime", facecolor="none", zorder=5)
    ax.add_patch(rect)
    ax.arrow(0, 0, c_len, 0, width=0.002 * roi_length, head_width=0.01 * roi_length,
             color="lime", length_includes_head=True, zorder=6)


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
            # row: [x, y, cosθ, sinθ, v_x, v_y, width, length, 1, 0, 0]
            x = row[0]
            y = row[1]
            cos_h = row[2]
            sin_h = row[3]
            w = row[6]
            l = row[7]
            # 빈 값(0,0,0,0,0,...) 은 건너뛴다
            # 간단히 norm(x,y,cosh,sinh,l,w)==0 -> skip
            if (abs(x) < 1e-6 and abs(y) < 1e-6 and abs(cos_h) < 1e-6 and abs(sin_h) < 1e-6):
                continue
            if (abs(l) < 1e-6 and abs(w) < 1e-6):
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
                             alpha=0.8)  # 0.5 투명도로 한다
            ax.add_patch(rect)

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
    observations = env.observations
    observation = observations["default_agent"]
    ego = env.vehicle
    # step 한 번
    action = [0.0, 0.1]
    iter_num = 332
    for i in range(iter_num):
        env.step(action)

    ego_x, ego_y = ego.position
    ego_yaw = ego.heading_theta
    roadnet = env.current_map.road_network

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    roi_len = observation.lane_roi_length

    # 1) 전체 RoadNetwork 표시 (흰색)
    visualize_entire_road_network(roadnet,
                                  ego_x,
                                  ego_y,
                                  ego_yaw,
                                  sampling_interval=1.0,
                                  roi_length=roi_len,
                                  ax=ax)

    # observation 정보 얻기
    observation_dict = observation.observe(ego)
    lanes_array = observation_dict["lanes_array"]           # (70, 20, 12)
    nav_lanes_array = observation_dict["nav_lanes_array"]   # (25, 20, 12)
    static_objects = observation_dict["static_objects"]      # (N, 10)
    neighbors_history = observation_dict["neighbors_history"]# (32, 21, 11)

    # 2) 차선/내비 차선 시각화
    visualize_lanes_array(lanes_array, length=roi_len, color="orange", ax=ax)
    visualize_lanes_array(nav_lanes_array, length=roi_len, color="red", ax=ax)
    # 3) 정적 객체
    visualize_static_objects(ax, static_objects, edgecolor="purple", fill=False)
    # 4) 이웃 vehicle 히스토리 시각화
    visualize_neighbors_history(ax, neighbors_history)

    ax.set_facecolor("black")
    ax.set_title("ROI Lanes + Static Objects + NeighborsHistory")
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.savefig("lanes_static_neigh_history.png", dpi=150, facecolor="black")
    plt.close()
    print("Saved lanes_static_neigh_history.png")

    env.close()


if __name__ == "__main__":
    main()
