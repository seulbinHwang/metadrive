import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Union
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.envs.diffusion_planner_env import DiffusionPlannerEnv


def extract_local_lanes_in_square_bbox(
        ego: BaseVehicle, road_network: NodeRoadNetwork, roi_length: float,
        max_lane_num: int, max_nav_lane_num: int,
        num_points_per_lane: int) -> Tuple[np.ndarray, np.ndarray]:
    checkpoint_node_ids = ego.navigation.checkpoints
    ego_x, ego_y = ego.position
    ego_yaw = ego.heading_theta

    half_len = roi_length * 0.5
    all_lanes = road_network.get_all_lanes()
    lanes_indices: List[Tuple[str, str, int]] = road_network.indices

    candidate_lanes: List[AbstractLane] = []
    candidate_lanes_indices: List[Tuple[str, str, int]] = []
    # --- polygon 검사 ---
    for lane, lane_indices in zip(all_lanes, lanes_indices):
        ln_len = lane.length
        # (1) 너무 짧으면 미리 패스
        if ln_len < 1e-6:
            continue
        poly = lane.polygon
        if poly is None or len(poly) == 0:
            continue
        poly = np.array(poly, dtype=np.float32)
        px = poly[:, 0]
        py = poly[:, 1]
        # 로컬 변환 (batch)
        lx, ly = _to_local_coords_batch(px, py, ego_x, ego_y, ego_yaw)

        min_x, max_x = lx.min(), lx.max()
        min_y, max_y = ly.min(), ly.max()
        # [-half_len, half_len] 범위 AABB 겹침
        if ((max_x>=-half_len) and (min_x<=half_len)) and \
                ((max_y>=-half_len) and (min_y<=half_len)):
            candidate_lanes.append(lane)
            candidate_lanes_indices.append(lane_indices)

    # --- 거리 정렬 ---
    ego_pos = np.array([ego_x, ego_y], dtype=float)

    def lane_dist(lan: AbstractLane):
        return lan.distance(ego_pos)

    # (2) candidate_lanes, candidate_lanes_indices를 같이 정렬
    cand_pairs = list(zip(candidate_lanes, candidate_lanes_indices))
    cand_pairs.sort(key=lambda pair: lane_dist(pair[0]))

    # 맥스개수 만큼 선택
    selected_pairs = cand_pairs[:max_lane_num]
    selected_lanes = [p[0] for p in selected_pairs]
    selected_indices = [p[1] for p in selected_pairs]

    # --- 결과 ---
    lanes_array = np.zeros((max_lane_num, num_points_per_lane, 12),
                           dtype=np.float32)
    nav_lanes_array = np.zeros((max_nav_lane_num, num_points_per_lane, 12),
                               dtype=np.float32)
    nav_idx = 0
    for idx, (ln,
              ln_idx_tuple) in enumerate(zip(selected_lanes, selected_indices)):
        ln_len = ln.length
        s_vals = np.linspace(0.0, ln_len, num_points_per_lane,
                             endpoint=True)  # shape=(num_points_per_lane,)
        w = getattr(ln, "width", 3.5)
        half_w = w * 0.5

        # (1) 월드 좌표 샘플링
        N = len(s_vals)
        center_w = np.zeros((N, 2), dtype=np.float32)
        for i, s_val in enumerate(s_vals):
            cx, cy = ln.position(s_val, 0.0)
            center_w[i] = (cx, cy)

        # (2) 로컬 변환 (batch)
        local_cx, local_cy = _to_local_coords_batch(center_w[:, 0], center_w[:,
                                                                             1],
                                                    ego_x, ego_y, ego_yaw)
        local_cxy = np.column_stack((local_cx, local_cy))  # shape=(N,2)
        # (3) 인접점 차이 np.diff (마지막=0)
        # diffs1 = np.diff(np.column_stack((local_cx, local_cy)), axis=0, prepend=np.array([[local_cx[0], local_cy[0]]]))

        diffs = local_cxy[1:] - local_cxy[:-1]  #
        diffs = np.insert(diffs, diffs.shape[0], 0,
                          axis=0)  # polyline_vector: (20, 2)

        diffs[-1] = 0.0  # 마지막=0
        # 저장
        lanes_array[idx, :, 0] = local_cx
        lanes_array[idx, :, 1] = local_cy
        lanes_array[idx, :, 2] = diffs[:, 0]
        lanes_array[idx, :, 3] = diffs[:, 1]

        # 방향(단위벡터)
        norms = np.hypot(diffs[:, 0], diffs[:, 1])
        dirs = np.zeros((N, 2), dtype=np.float32)

        # (a) 마스크로 길이가 매우 작은 부분 처리
        mask_small = (norms < 1e-6)
        mask_large = ~mask_small

        # (b) 길이가 충분히 크면 diffs / norms
        dirs[mask_large] = diffs[mask_large] / norms[mask_large, None]
        # (c) 너무 짧은 곳은 heading_theta_at() → local 회전
        small_idxs = np.nonzero(mask_small)[0]
        if len(small_idxs) > 0:
            heading_ths = np.array(
                [ln.heading_theta_at(s_vals[i]) for i in small_idxs],
                dtype=float)
            dir_x = np.cos(heading_ths)
            dir_y = np.sin(heading_ths)

            loc_dirx, loc_diry = _rotate_vectors_batch(dir_x, dir_y, ego_yaw)
            dirs[small_idxs, 0] = loc_dirx
            dirs[small_idxs, 1] = loc_diry

        # 왼/오른 offset
        left_off = np.zeros((N, 2), dtype=np.float32)
        left_off[:, 0] = -dirs[:, 1] * half_w
        left_off[:, 1] = dirs[:, 0] * half_w
        right_off = -left_off

        lanes_array[idx, :, 4] = left_off[:, 0]
        lanes_array[idx, :, 5] = left_off[:, 1]
        lanes_array[idx, :, 6] = right_off[:, 0]
        lanes_array[idx, :, 7] = right_off[:, 1]
        # 신호원핫
        lanes_array[idx, :, 8] = 1.0  # [1,0,0,0]
        start_node_str = ln_idx_tuple[0]
        end_node_str = ln_idx_tuple[1]
        if nav_idx < max_nav_lane_num:
            if (start_node_str
                    in checkpoint_node_ids) and (end_node_str
                                                 in checkpoint_node_ids):
                nav_lanes_array[nav_idx] = lanes_array[idx]
                nav_idx += 1
    return lanes_array, nav_lanes_array


def _to_local_coords_batch(px: np.ndarray, py: np.ndarray, ego_x: float,
                           ego_y: float,
                           ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    px, py: shape (N,) 전역 좌표배열
    자차(Ego)의 (ego_x, ego_y, ego_yaw)를 기준으로 로컬 변환을 벡터 연산으로 수행.
    """
    dx = px - ego_x
    dy = py - ego_y
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    # local_x = dx*c + dy*s
    # local_y = -dx*s + dy*c
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return local_x, local_y


def _rotate_vectors_batch(vx: np.ndarray, vy: np.ndarray,
                          ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    순수 벡터(vx, vy)를 자차 yaw만큼 회전. 평행이동은 없음 (순수 방향벡터).
    shape(N,) -> shape(N,)
    """
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    # x' = x*c + y*s
    # y' = -x*s + y*c
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
    """
    전체 RoadNetwork를 자차 로컬 좌표계로 변환(벡터 연산)하여 흰색으로 시각화 (점선=중앙, 실선=양끝)
    """
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

        # s sampling
        s_vals = np.arange(0.0, ln_len + 1e-9, sampling_interval)
        if len(s_vals) < 2:
            s_vals = np.array([0.0, ln_len])

        # (1) 세계 좌표에서 center/left/right
        N = len(s_vals)
        center_w = np.zeros((N, 2), dtype=np.float32)
        left_w = np.zeros((N, 2), dtype=np.float32)
        right_w = np.zeros((N, 2), dtype=np.float32)
        # lane.position은 for문
        for i, s_val in enumerate(s_vals):
            cx, cy = lane.position(s_val, 0.0)
            lx, ly = lane.position(s_val, +half_w)
            rx, ry = lane.position(s_val, -half_w)
            center_w[i] = (cx, cy)
            left_w[i] = (lx, ly)
            right_w[i] = (rx, ry)

        # (2) 자차 로컬로 변환 (벡터 연산)
        cx_l, cy_l = _to_local_coords_batch(center_w[:, 0], center_w[:, 1],
                                            ego_x, ego_y, ego_yaw)
        lx_l, ly_l = _to_local_coords_batch(left_w[:, 0], left_w[:, 1], ego_x,
                                            ego_y, ego_yaw)
        rx_l, ry_l = _to_local_coords_batch(right_w[:, 0], right_w[:, 1], ego_x,
                                            ego_y, ego_yaw)

        ax.plot(cx_l, cy_l, "--", color="white", linewidth=1.0)
        ax.plot(lx_l, ly_l, "-", color="white", linewidth=1.2)
        ax.plot(rx_l, ry_l, "-", color="white", linewidth=1.2)

    ax.set_facecolor("black")
    # 자차 표시
    c_len = roi_length / 20.0
    c_wid = roi_length / 100.0
    rx = -c_len * 0.5
    ry = -c_wid * 0.5
    rect = Rectangle((rx, ry),
                     c_len,
                     c_wid,
                     edgecolor="lime",
                     facecolor="none",
                     zorder=5)
    ax.add_patch(rect)
    ax.arrow(0,
             0,
             c_len,
             0,
             width=0.002 * roi_length,
             head_width=0.01 * roi_length,
             color="lime",
             length_includes_head=True,
             zorder=6)


def visualize_lanes_array(lanes_array: np.ndarray,
                          length: float,
                          color="orange",
                          ax=None):
    """
    ROI 내 차선 (주황) 표시
    """
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

    # # 자차
    # car_len = length/20.0
    # car_wid = length/100.0
    # rect = Rectangle(
    #     xy=(-car_len/2.0, -car_wid/2.0),
    #     width=car_len, height=car_wid,
    #     edgecolor="black", facecolor="none", zorder=10
    # )
    # ax.add_patch(rect)
    # ax.arrow(
    #     0,0, car_len,0,
    #     width=0.002*length,
    #     head_width=0.01*length,
    #     color="black", length_includes_head=True, zorder=11
    # )
    # --- 정사각형 테두리 그리기 ---
    # 원점 중심으로 가로 200m, 세로 200m = x, y각각 [-100, +100]
    square = Rectangle(
        (-length//2, -length//2),  # 좌측 하단 좌표
        length,  # 너비 (100 - (-100))
        length,  # 높이 (100 - (-100))
        linewidth=2,
        edgecolor='cyan',
        facecolor='none')
    ax.add_patch(square)

def visualize_static_objects(ax,
                             static_objects: np.ndarray,
                             edgecolor="purple",
                             fill=False):
    """
    정적 객체 (5,10): [ x, y, cosθ, sinθ, width, length, ... one_hot(4) ]을
    사각형(Rectangle)으로 표시. 색상=보라색
    """
    # static_objects.shape = (N, 10)
    # row = [x, y, cosθ, sinθ, width, length, one_hot(4)]
    N = static_objects.shape[0]
    for i in range(N):
        row = static_objects[i]
        x, y     = row[0], row[1]
        cos_h    = row[2]
        sin_h    = row[3]
        w        = row[4]
        l        = row[5]
        heading  = math.atan2(sin_h, cos_h)  # 라디안
        heading_deg = math.degrees(heading)

        # matplotlib Rectangle은 "왼쪽 하단"을 anchor로 하여 그려지므로,
        # 중심 (x,y)를 anchor로 만들려면, anchor_x/y를 다음과 같이 조정:
        anchor_x = x - l/2.0
        anchor_y = y - w/2.0

        rect = Rectangle(
            (anchor_x, anchor_y),
            l,
            w,
            angle=heading_deg,
            edgecolor=edgecolor,
            facecolor=edgecolor,
            lw=1.5,
            zorder=10,
            alpha=1.0
        )
        ax.add_patch(rect)

def main():
    config = {
        "num_scenarios": 1,  # 사용할 랜덤 맵 수 (예: 1000개 맵)
        "start_seed": 0,  # 시드 시작값 (0번부터 순차 생성)
        "traffic_density": 0.1,  # 교통량 밀도 (기본값 0.1)
        "map": 5,
        # "discrete_action": False,  # 연속 행동 사용 (False가 기본값)
        # 추가 필요 설정이 있다면 이곳에 작성
        "traffic_mode": "trigger",
        "random_traffic": True,
        "use_render": False,
    }

    env = DiffusionPlannerEnv(config)
    env.reset()
    observations = env.observations
    observation = observations["default_agent"]

    ego = env.vehicle
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

    # # 2) ROI 내 차선(배치연산) 추출
    # import time
    # start_time = time.time()
    # # ['>', '>>', '>>>', '1C0_0_', '1C0_1_', '2C0_0_', '2C0_1_', '3C0_0_', '3C0_1_', '4X0_0_', '4X0_1_', '5C0_0_', '5C0_1_']
    # roadnet = env.current_map.road_network
    # lanes_array, nav_lanes_array = extract_local_lanes_in_square_bbox(
    #     ego,
    #     roadnet,
    #     roi_length=roi_len,
    #     max_lane_num=70,
    #     max_nav_lane_num=25,
    #     num_points_per_lane=20)
    # print(f"elapsed time: {time.time()-start_time:.3f} sec")
    observation_dict = observation.observe(ego)
    lanes_array = observation_dict["lanes_array"] # (70, 20, 12)
    nav_lanes_array = observation_dict["nav_lanes_array"] # (25, 20, 12)
    """ static_objects
    - 5: number of static objects
    - 10: number of features
        - x, y, cos(heading), sin(heading), width, length, one_hot(CZONE_SIGN, BARRIER, TRAFFIC_CONE, GENERIC_OBJECT)]
    
    """
    static_objects = observation_dict["static_objects"] # (5, 10)
    # 3) ROI 차선(오렌지) 표시
    visualize_lanes_array(lanes_array, length=roi_len, color="orange", ax=ax)
    visualize_lanes_array(nav_lanes_array, length=roi_len, color="red", ax=ax)


    # 2) 정적 객체 시각화
    visualize_static_objects(ax, static_objects, edgecolor="purple", fill=False)

    ax.set_facecolor("black")
    ax.set_title("ROI Lanes + Static Objects")
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.savefig("lanes_and_static_objects.png", dpi=150, facecolor="black")
    plt.close()
    print("Saved lanes_and_static_objects.png")

    env.close()


if __name__ == "__main__":
    main()
