import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib.patches import Rectangle

def _to_local_coords(
    x_world: float,
    y_world: float,
    ego_x: float,
    ego_y: float,
    ego_yaw: float
) -> Tuple[float, float]:
    """
    전역 좌표(x_world, y_world)를 자차(Ego) 좌표계로 변환.
    (ego_x, ego_y, ego_yaw)를 원점(0,0), heading=0 으로 맞춤.
    """
    dx = x_world - ego_x
    dy = y_world - ego_y
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)

    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return (local_x, local_y)

def visualize_entire_road_network(
    road_network,
    checkpoint_node_ids: List[str],
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    sampling_interval: float = 1.0,
    out_file: str = "road_network.png",
    roi_length: float = 200.0
) -> None:
    """
    NodeRoadNetwork 기반 맵 전체를 자차(Ego) 좌표계 기준으로 그려서 저장한다.

    Args:
        road_network: NodeRoadNetwork 객체
        ego_x, ego_y, ego_yaw: 자차(Ego) 위치/방향 (전역)
        sampling_interval: 차선을 샘플링할 간격(미터)
        out_file: 저장할 이미지 파일명
        roi_length: 자차 주변 시각화 영역 크기 (차량 크기 결정 용도)
    """
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    # 1) road_network에서 모든 lane 추출
    # road_network: NodeRoadNetwork
    all_lanes = road_network.get_all_lanes()  # List[AbstractLane]
    lanes_indices:List[Tuple[str, str, int]] = road_network.indices

    # 2) 각 lane에 대해 중앙선, 왼경계, 오른경계를 자차 로컬로 변환해 그림
    for lane , lane_index in zip(all_lanes, lanes_indices):
        start_node_str = lane_index[0]
        end_node_str = lane_index[1]
        if start_node_str in checkpoint_node_ids and end_node_str in checkpoint_node_ids:
            color = "red"
        else:
            color = "white"
        # lane length, width
        lane_length = lane.length
        if lane_length < 1e-6:
            continue
        lane_width = getattr(lane, "width", 3.5)

        # 종방향 s = 0 ~ lane_length, sampling_interval씩
        s_vals = np.arange(0.0, lane_length + 1e-6, sampling_interval)
        if len(s_vals) < 2:
            # 최소 2개는 있어야 plot 가능
            s_vals = np.array([0.0, lane_length])

        # 중앙선 (자차 좌표)
        center_local_x = []
        center_local_y = []
        # 왼경계
        left_local_x = []
        left_local_y = []
        # 오른경계
        right_local_x = []
        right_local_y = []

        half_w = 0.5 * lane_width
        for s in s_vals:
            # 중앙선 world
            cx, cy = lane.position(s, 0.0)
            lx, ly = lane.position(s, +half_w)
            rx, ry = lane.position(s, -half_w)

            # 자차 좌표계 변환
            c_local = _to_local_coords(cx, cy, ego_x, ego_y, ego_yaw)
            l_local = _to_local_coords(lx, ly, ego_x, ego_y, ego_yaw)
            r_local = _to_local_coords(rx, ry, ego_x, ego_y, ego_yaw)

            center_local_x.append(c_local[0])
            center_local_y.append(c_local[1])
            left_local_x.append(l_local[0])
            left_local_y.append(l_local[1])
            right_local_x.append(r_local[0])
            right_local_y.append(r_local[1])

        # 3) matplotlib에 그리기
        #   중앙선: 흰 점선, 양끝선: 흰 실선
        plt.plot(center_local_x, center_local_y, "--", color=color, linewidth=1.0)
        plt.plot(left_local_x,   left_local_y,   "-",  color=color, linewidth=1.2)
        plt.plot(right_local_x,  right_local_y,  "-",  color=color, linewidth=1.2)

    # 4) 자차(0,0) 에 차량 표시 (직사각형)
    car_length = roi_length / 20.0
    car_width  = roi_length / 100.0
    rect_x = -car_length * 0.5
    rect_y = -car_width  * 0.5

    # 흰 바탕에 흰 선은 안 보이므로, 배경 설정(예: 검은 배경)
    ax.set_facecolor("black")

    # 자차 rectangle
    rect = Rectangle(
        (rect_x, rect_y), car_length, car_width,
        edgecolor="lime", facecolor="none", zorder=5
    )
    ax.add_patch(rect)

    # heading 화살표 : +x방향
    plt.arrow(
        0, 0, car_length, 0,
        width=0.002*roi_length,
        head_width=0.01*roi_length,
        color="lime",
        length_includes_head=True,
        zorder=6
    )

    # 기타 설정
    plt.title("Road Network (Ego Coord)")
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.axis("equal")

    # 이미지 저장
    plt.savefig(out_file, dpi=150, facecolor="black")  # 검은 배경
    plt.close()
    print(f"[visualize_entire_road_network] Saved image to {out_file}")

if __name__ == "__main__":
    # (예) MetaDriveEnv 초기화
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

    # 자차 위치
    ego_vehicle = env.vehicle  # single-agent 가정
    ego_x, ego_y = ego_vehicle.position
    ego_yaw = ego_vehicle.heading_theta
    # ['>', '>>', '>>>', '1C0_0_', '1C0_1_', '2C0_0_', '2C0_1_', '3C0_0_', '3C0_1_', '4X0_0_', '4X0_1_', '5C0_0_', '5C0_1_']
    checkpoint_node_ids = ego_vehicle.navigation.checkpoints
    # road network
    """
    env.current_map: PGMap
    env.current_map.road_network: NodeRoadNetwork
    """
    road_network = env.current_map.road_network


    # 시각화
    visualize_entire_road_network(
        road_network=road_network,
        checkpoint_node_ids = checkpoint_node_ids,
        ego_x=ego_x, ego_y=ego_y, ego_yaw=ego_yaw,
        sampling_interval=1.0,
        out_file="road_network_ego.png",
        roi_length=200.0
    )

    env.close()
