import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Union
import colorsys

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.envs.diffusion_planner_env import DiffusionPlannerEnv
import time
import numpy as np
import math
# ── utils_vis.py ────────────────────────────────────────────
import numpy as np

RED = (1, 0, 0, 1)          # RGBA
def draw_ref_path_points(env,
                         ego: BaseVehicle,
                         ref_path: np.ndarray,
                         dir_len: float = 0.8,     # 헤딩 화살표 길이 [m]
                         z_offset: float = 0.55,
                         line_width: int = 15):
    """
    ref_path : (N,4) [x_local, y_local, cos_h, sin_h] (Ego 좌표계)
    각 점에서 헤딩 방향(로컬 cos/sin)을 월드로 변환한 선분을 그림.
    """
    # ── 1) 지난 프레임 객체 지우기 ───────────────────────
    if hasattr(env.engine, "_ref_np_list"):
        for np_ in env.engine._ref_np_list:
            np_.removeNode()
    env.engine._ref_np_list = []

    # ── 2) Ego 자세 값 꺼내기 ────────────────────────────
    ego_x, ego_y       = ego.rear_axle_xy
    yaw_c, yaw_s       = ego.heading                 # cosθ, sinθ
    ego_z              = ego.get_z()

    for x_l, y_l, c_h, s_h in ref_path:
        # (a) 로컬 → 월드 위치 변환
        wx = ego_x + x_l * yaw_c - y_l * yaw_s
        wy = ego_y + x_l * yaw_s + y_l * yaw_c
        wz = ego_z + z_offset

        # (b) 로컬 헤딩 → 월드 방향 벡터 변환
        vx = c_h * yaw_c - s_h * yaw_s
        vy = c_h * yaw_s + s_h * yaw_c

        # (c) 선분 끝점 = 시작점 + dir_len * (vx,vy)
        ex = wx + dir_len * vx
        ey = wy + dir_len * vy
        ez = wz                    # 동일 높이

        # (d) 3D 선분 그리기
        np_line = env.engine._draw_line_3d(
            [wx, wy, wz],
            [ex, ey, ez],
            RED,
            line_width
        )
        np_line.reparentTo(env.engine.render)
        env.engine._ref_np_list.append(np_line)


def extract_centerline_in_ego_frame(lane,
                                    ego_position_world: np.ndarray,
                                    ego_heading_world: float,
                                    step: float = 0.5,
                                    M_max: int = 80) -> np.ndarray:
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
    s_current = max(s_ego + step, 0.0)
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
                           ego_y: float,
                           ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    dx = px - ego_x
    dy = py - ego_y
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return local_x, local_y


def _rotate_vectors_batch(vx: np.ndarray, vy: np.ndarray,
                          ego_yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    x2 = vx * c + vy * s
    y2 = -vx * s + vy * c
    return x2, y2


def get_ref_path(ego: BaseVehicle,
                 step_: float = 1.,
                 current_lane_idx: int = 1) -> np.ndarray:
    """
    반환값: (max_point, 4)  =  [x_local, y_local, cos_h, sin_h]
    """
    max_point = 80

    # ── 1) 현재 / 다음 기준 차선 구하기 ───────────────────────────
    cur_lane   = ego.navigation.current_lane
    ref_lanes  = ego.navigation.current_ref_lanes          # type: List[AbstractLane]
    lane_id    = ref_lanes.index(cur_lane)                 # 현재 차선이 ref_lanes 중 몇 번째인지
    lane_id = 1
    a_cur_lane = ref_lanes[lane_id]                      # 현재 차선
    next_lanes = ego.navigation.next_ref_lanes
    next_lane  = next_lanes[lane_id] if next_lanes else cur_lane

    # ── 2) ego 자세 꺼내기 ───────────────────────────────────────
    ego_x, ego_y = ego.rear_axle_xy
    ego_yaw      = ego.heading_theta

    # ── 3) 기준 차선 1 · 2 에서 경로 점 추출 ─────────────────────
    ref1 = extract_centerline_in_ego_frame(
        a_cur_lane, np.array([ego_x, ego_y]), ego_yaw,
        step=step_, M_max=max_point
    )
    remain = max_point - ref1.shape[0]

    ref2 = np.zeros((0, 4), np.float32)
    if remain > 0:
        ref2 = extract_centerline_in_ego_frame(
            next_lane, np.array([ego_x, ego_y]), ego_yaw,
            step=step_, M_max=remain
        )

    ref_path = np.concatenate([ref1, ref2], axis=0)

    # # ── 4) ego → 첫 점 사이 보간점 삽입 ───────────────────────────
    # if ref_path.shape[0] > 0:
    #     first_pt = ref_path[0]                       # [x_l, y_l, cos, sin]
    #     dist     = math.hypot(first_pt[0], first_pt[1])
    #     n_insert = int(dist // step_)                # 넣어야 할 점 개수
    #
    #     if n_insert > 0:
    #         # 방향 단위 벡터 (ego 로컬)
    #         dir_vec = np.array(first_pt[:2]) / dist
    #         ins_list = []
    #         for k in range(n_insert):
    #             d = step_ * (k + 1)
    #             if d >= dist:        # 첫 점과 겹치지 않도록
    #                 break
    #             xy = dir_vec * d
    #             yaw_local = math.atan2(dir_vec[1], dir_vec[0])
    #             ins_list.append([xy[0], xy[1],
    #                              math.cos(yaw_local), math.sin(yaw_local)])
    #
    #         if ins_list:
    #             ref_path = np.vstack([np.array(ins_list, np.float32),
    #                                   ref_path])

    # ── 5) 길이 맞추기 (앞에서 늘렸으니 뒤에서 자름) ────────────────
    if ref_path.shape[0] > max_point:
        ref_path = ref_path[:max_point]
    else:
        # 부족하면 마지막 점 복제
        pad = max_point - ref_path.shape[0]
        if pad > 0 and ref_path.shape[0] > 0:
            last = np.tile(ref_path[-1], (pad, 1))
            ref_path = np.vstack([ref_path, last])

    return ref_path.astype(np.float32)


def main():

    config = {
        "num_scenarios": 1,
        "start_seed": 0,
        "map": 5,
        "random_traffic": True,
        "use_render": True,  # <- 화면 창 활성화
        "debug": True,
    }

    env = DiffusionPlannerEnv(config)
    env.reset()
    ego: BaseVehicle = env.vehicle  # 최신 ego
    done = False
    while not done:
        ref_path = get_ref_path(ego)  # 매 스텝마다 새로운 목표 경로

        # MetaDrive(Gym) 규격: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(ref_path)
        done = terminated or truncated
        # draw_ref_path_points(env, ego, ref_path)
        env.engine.taskMgr.step()                          # 3) 한 프레임 그리기

        # 사람이 보기 편하도록 살짝 쉬어 줍니다 (fps ≈ 60)
        # time.sleep(env.engine.global_config["physics_world_step_size"])
    env.close()


if __name__ == "__main__":
    main()
