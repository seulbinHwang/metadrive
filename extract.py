import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Union

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.lane.abs_lane import AbstractLane

def _to_local_coord(
    x_world: float,
    y_world: float,
    ego_x: float,
    ego_y: float,
    ego_yaw: float
) -> Tuple[float, float]:
    """
    전역 좌표 (x_world, y_world)를 자차 로컬 좌표계로 변환.
    자차가 (ego_x, ego_y), heading=ego_yaw 라고 할 때,
     local_x = (dx)*cos(ego_yaw) + (dy)*sin(ego_yaw)
     local_y = -(dx)*sin(ego_yaw) + (dy)*cos(ego_yaw)
    """
    dx = x_world - ego_x
    dy = y_world - ego_y
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    local_x = dx*c + dy*s
    local_y = -dx*s + dy*c
    return (local_x, local_y)

def extract_local_lanes_in_square_bbox(
    road_network: NodeRoadNetwork,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    length: float,
    max_lane_num: int,
    num_points_per_lane: int
) -> np.ndarray:
    """
    NodeRoadNetwork로부터, ego 좌표계 기준 (0,0) 중심, 길이 length인 정사각형 영역 내 차선을 추출.
    차선은 자차와의 거리 순으로 max_lane_num 개까지만.
    각 차선은 num_points_per_lane로 샘플링한 (중앙선) 정보를 포함해 shape = (max_lane_num, num_points_per_lane, 12) 를 만든다.

    12차원 구조:
     [0-1]: (local_x, local_y) - 차선 중앙선(로컬좌표)
     [2-3]: (dx, dy) - 인접 포인트 차이(마지막은 (0,0))
     [4-5]: (dx, dy) - 왼쪽 경계 오프셋
     [6-7]: (dx, dy) - 오른쪽 경계 오프셋
     [8-11]: 교통신호 원-핫(예: [1,0,0,0])
    """
    half_len = length * 0.5
    # 1) road_network.get_all_lanes() -> List[AbstractLane]
    all_lanes: List[AbstractLane] = road_network.get_all_lanes()

    # 2) 정사각형 범위와 겹치는 lane 필터링
    #    polygon bounding box -> 자차 로컬로 변환 -> [-half_len, half_len]와 겹치면 포함
    candidate_lanes: List[AbstractLane] = []
    for lane in all_lanes:
        poly = lane.polygon  # (N,2) np.array
        if poly is None or len(poly) == 0:
            continue

        # 로컬로 변환
        local_x_vals = []
        local_y_vals = []
        for px, py in poly:
            lx, ly = _to_local_coord(px, py, ego_x, ego_y, ego_yaw)
            local_x_vals.append(lx)
            local_y_vals.append(ly)

        lxmin, lxmax = min(local_x_vals), max(local_x_vals)
        lymin, lymax = min(local_y_vals), max(local_y_vals)

        # [-half_len, half_len] 범위와 AABB 겹침 체크
        if ((lxmax >= -half_len) or (lxmin <= half_len)) and \
                ((lymax >= -half_len) or (lymin <= half_len)):
            candidate_lanes.append(lane)

    # 3) 자차(전역 coords)와의 최소 거리로 정렬
    def lane_distance(l: AbstractLane, ego_pos: np.ndarray) -> float:
        return l.distance(ego_pos)
    ego_pos = np.array([ego_x, ego_y], dtype=float)
    candidate_lanes.sort(key=lambda l: lane_distance(l, ego_pos))

    # 4) 상위 max_lane_num 개만
    selected_lanes = candidate_lanes[:max_lane_num]

    # 5) 결과 (max_lane_num, num_points_per_lane, 12)
    lanes_array = np.zeros((max_lane_num, num_points_per_lane, 12), dtype=np.float32)

    # 6) 각 차선에 대해 샘플링
    for idx, lane in enumerate(selected_lanes):
        lane_length = lane.length
        if lane_length < 1e-6:
            continue

        # 등간격 s
        s_vals = np.linspace(0.0, lane_length, num_points_per_lane, endpoint=True)

        lane_width = getattr(lane, "width", 3.5)
        half_w = lane_width * 0.5

        prev_lx, prev_ly = None, None
        for i, s_val in enumerate(s_vals):
            # 중앙선(전역)
            cx, cy = lane.position(s_val, 0.0)
            # 로컬 변환
            lx, ly = _to_local_coord(cx, cy, ego_x, ego_y, ego_yaw)
            lanes_array[idx, i, 0] = lx
            lanes_array[idx, i, 1] = ly

            # 인접점 차이
            if i == 0:
                dx, dy = 0.0, 0.0
            else:
                dx = lx - prev_lx
                dy = ly - prev_ly
            if i == (num_points_per_lane - 1):
                dx, dy = 0.0, 0.0
            lanes_array[idx, i, 2] = dx
            lanes_array[idx, i, 3] = dy

            # 접선 단위벡터
            mag = math.hypot(dx, dy)
            if mag < 1e-6:
                # backup: heading_theta_at
                heading_th = lane.heading_theta_at(s_val)
                dir_x = math.cos(heading_th)
                dir_y = math.sin(heading_th)
                # 로컬 좌표로 회전:
                #   (dir_x, dir_y) in world -> local
                #   여기서는 차분을 직접 world->local로 변환해도 되지만,
                #   단순히 "이미 local"으로 lx,ly를 구했으니 이 부분은 간단 처리.
                #   => (dx,dy)=0이면 오프셋은 그냥 heading을 이용
                #   => heading_th기준 왼/오른 경계 구할 수 있지만, 간단히 dir_x,dir_y만...
            else:
                dir_x = dx / mag
                dir_y = dy / mag

            # 왼/오른 오프셋(자차 로컬)
            left_dx = -dir_y * half_w
            left_dy =  dir_x * half_w
            right_dx=  dir_y * half_w
            right_dy= -dir_x * half_w
            lanes_array[idx, i, 4] = left_dx
            lanes_array[idx, i, 5] = left_dy
            lanes_array[idx, i, 6] = right_dx
            lanes_array[idx, i, 7] = right_dy

            # 교통 신호 원핫
            lanes_array[idx, i, 8:12] = [1, 0, 0, 0]

            prev_lx, prev_ly = lx, ly

    return lanes_array

def visualize_lanes_array(
    lanes_array: np.ndarray,
    length: float,
    out_file: str = "lanes_array.png"
) -> None:
    """
    lanes_array 시각화.
    - 0-1: 로컬 중심선 점(점선)
    - 4-5, 6-7: 왼/오른 경계 실선 (주황색)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.set_aspect("equal")

    max_lane_num = lanes_array.shape[0]

    for lane_idx in range(max_lane_num):
        lane_data = lanes_array[lane_idx]  # (num_points_per_lane,12)

        # 중앙선
        center_xy = lane_data[:, 0:2]  # (x,y)
        left_off  = lane_data[:, 4:6]
        right_off = lane_data[:, 6:8]

        left_xy  = center_xy + left_off
        right_xy = center_xy + right_off

        # 중앙: 점선(주황)
        plt.plot(center_xy[:,0], center_xy[:,1], "--", color="orange")
        # 양끝: 실선(주황)
        plt.plot(left_xy[:,0], left_xy[:,1], "-", color="orange")
        plt.plot(right_xy[:,0], right_xy[:,1], "-", color="orange")

    # 자차
    car_len = length / 20.0
    car_wid = length / 100.0
    rect = Rectangle(
        xy=(-car_len/2.0, -car_wid/2.0),
        width=car_len, height=car_wid,
        edgecolor="black", facecolor="none", zorder=10
    )
    ax.add_patch(rect)
    # heading = +x
    plt.arrow(
        0, 0, car_len, 0,
        width=0.002*length,
        head_width=0.01*length,
        color="black", length_includes_head=True, zorder=11
    )

    plt.title("Local Lane Visualization")
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.axis("equal")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved lane array image to {out_file}")

if __name__ == "__main__":
    # === 예시: MetaDrive 사용 ===
    from metadrive import MetaDriveEnv


    default_config = MetaDriveEnv.default_config()
    default_config.update(
        dict(
            use_render=False,
            num_scenarios=1,  # environment_num 대신 사용
            start_seed=0,
            map=5,
            traffic_density=0.1))
    env = MetaDriveEnv(default_config)
    env.reset()

    # step 한 번
    action = [0.0, 0.1]
    env.step(action)

    ego = env.vehicle
    ego_x, ego_y = ego.position
    ego_yaw = ego.heading_theta
    roadnet = env.current_map.road_network  # NodeRoadNetwork

    # lanes_array
    lanes_array = extract_local_lanes_in_square_bbox(
        road_network=roadnet,
        ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw,
        length=200.0,
        max_lane_num=35,
        num_points_per_lane=20
    )

    # 그림 저장
    visualize_lanes_array(lanes_array, length=200.0, out_file="test_lane.png")

    env.close()
