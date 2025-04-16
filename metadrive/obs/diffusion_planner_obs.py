import gymnasium as gym

from metadrive.obs.observation_base import BaseObservation
from typing import Dict
import math
import numpy as np
from typing import List, Tuple, Union
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.lane.abs_lane import AbstractLane

# 이미 제공된 extract_local_lanes_in_square_bbox 함수
# (여기서는 코드를 그대로 붙여넣었다고 가정)
# from .somewhere import extract_local_lanes_in_square_bbox


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
    # 1) 우선 ego가 lane 상에서 어느 지점에 있는지(즉, s_ego)를 찾음
    #    - local_coordinates() 이용
    s_ego, r_ego = lane.local_coordinates(ego_position_world)
    #  (여기서 r_ego>0 이면 lane의 왼쪽 / r_ego<0 이면 lane의 오른쪽에 있음.)
    #  (관심있는 건 “lane상 앞쪽”이므로, s_ego보다 큰 s값 구간만 사용)
    #  s_ego가 lane.length 이상이면 이미 lane 끝을 넘어간 것이므로 반환값 없음

    # 2) s_ego ~ lane.length 구간을 step간격으로 샘플링
    s_list = []
    # s_ego보다 약간 앞에서부터 시작(필요하면 0.1m 등 추가 오프셋 가능)
    s_current = max(s_ego, 0.0)
    while s_current <= lane.length and len(s_list) < M_max:
        s_list.append(s_current)
        s_current += step
    if not s_list:
        # lane 앞부분이 없으면 곧바로 빈 배열 반환
        return np.zeros((0, 4), dtype=np.float32)

    # 3) 각 s에 대해 world 좌표계에서의 (x,y)와 yaw 구하고, -> Ego 좌표계로 변환
    #    - lane.position(s, 0) : 중심선이므로 lateral=0
    #    - lane.heading_theta_at(s)
    #    - Ego 좌표계 변환 => (x_local, y_local, heading_local)
    #      heading_local = (yaw_world - ego_heading_world)
    #      2D local pos는 아래처럼
    #         dx = x_world - ego_x
    #         dy = y_world - ego_y
    #         # Ego heading = ego_heading_world
    #         x_local =  dx*cos(ego_h) + dy*sin(ego_h)
    #         y_local = -dx*sin(ego_h) + dy*cos(ego_h)
    #      (또는 다른 관용적 방식 사용해도 OK)

    out_list = []
    cos_ego = math.cos(ego_heading_world)
    sin_ego = math.sin(ego_heading_world)

    for s in s_list:
        # (1) lane 상 world 좌표
        world_xy = lane.position(s, 0)  # np.array([x,y])
        world_yaw = lane.heading_theta_at(s)

        # (2) world -> ego 변환 (x_local,y_local)
        dx = world_xy[0] - ego_position_world[0]
        dy = world_xy[1] - ego_position_world[1]
        # ego 좌표계에서 x축은 ego_heading_world 방향이 “정면”
        #   x_local =  dx*cos(ego_h) + dy*sin(ego_h)
        #   y_local = -dx*sin(ego_h) + dy*cos(ego_h)
        x_local = dx * cos_ego + dy * sin_ego
        y_local = -dx * sin_ego + dy * cos_ego

        # (3) yaw_local
        #     = (world_yaw - ego_heading_world) 을 -π~+π 범위로 정규화
        yaw_local = world_yaw - ego_heading_world
        # 보통 wrap
        yaw_local = (yaw_local + math.pi) % (2 * math.pi) - math.pi

        # (4) cos(yaw_local), sin(yaw_local)
        c_h = math.cos(yaw_local)
        s_h = math.sin(yaw_local)

        out_list.append((x_local, y_local, c_h, s_h))

    return np.array(out_list, dtype=np.float32)

def extract_local_lanes_in_square_bbox(
        ego: BaseVehicle, lane_roi_length: float,
        max_lane_num: int, max_route_num: int,
        lane_len: int) -> Tuple[np.ndarray, np.ndarray]:
    road_network = ego.engine.current_map.road_network
    checkpoint_node_ids = ego.navigation.checkpoints
    ego_x, ego_y = ego.position
    ego_yaw = ego.heading_theta

    half_len = lane_roi_length * 0.5
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
    lanes_array = np.zeros((max_lane_num, lane_len, 12),
                           dtype=np.float32)
    nav_lanes_array = np.zeros((max_route_num, lane_len, 12),
                               dtype=np.float32)
    nav_idx = 0
    for idx, (ln,
              ln_idx_tuple) in enumerate(zip(selected_lanes, selected_indices)):
        ln_len = ln.length
        s_vals = np.linspace(0.0, ln_len, lane_len,
                             endpoint=True)  # shape=(lane_len,)
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
        if nav_idx < max_route_num:
            if (start_node_str
                    in checkpoint_node_ids) and (end_node_str
                                                 in checkpoint_node_ids):
                nav_lanes_array[nav_idx] = lanes_array[idx]
                nav_idx += 1
    return lanes_array, nav_lanes_array


class DiffusionPlannerObservation(BaseObservation):
    """
    이 클래스는 extract_local_lanes_in_square_bbox()를 호출해,
    lanes_array와 nav_lanes_array 두 가지 np.ndarray 정보를 Dict로 묶어 반환한다.
    """

    def __init__(self, config):
        super().__init__(config)
        self.observation_normalizer = config.observation_normalizer
        # 예: config 내부에 아래 key들이 있다고 가정
        # "lane_roi_length", "max_lane_num", "max_route_num", "lane_len"
        self.lane_roi_length = self.config.get("lane_roi_length", 200.0)
        self.max_lane_num = self.config.get("max_lane_num", 70)
        self.max_route_num = self.config.get("max_route_num", 25)
        self.lane_len = self.config.get("lane_len", 20)

        # lanes_array shape: (max_lane_num, lane_len, 12)
        # nav_lanes_array shape: (max_route_num, lane_len, 12)
        # dtype=float32, 값 범위는 특정 제한 없으므로 Box(-inf, inf, ... ) etc.
        # 필요 시에는 관습적으로 아주 넓은 범위를 clip해도 됨
        self._observation_space = gym.spaces.Dict({
            "lanes_array":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_lane_num,
                                      self.lane_len, 12),
                               dtype=np.float32),
            "nav_lanes_array":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_route_num,
                                      self.lane_len, 12),
                               dtype=np.float32),
        })

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    def _get_lanes(self, vehicle: BaseVehicle) -> Tuple[np.ndarray, np.ndarray]:
        # 실제 함수 호출
        lanes_array, nav_lanes_array = extract_local_lanes_in_square_bbox(
            ego=vehicle,
            lane_roi_length=self.lane_roi_length,
            max_lane_num=self.max_lane_num,
            max_route_num=self.max_route_num,
            lane_len=self.lane_len)
        return lanes_array, nav_lanes_array

    def _get_static_objects(self, vehicle: BaseVehicle) -> np.ndarray:
        object_manager = vehicle.engine.managers["object_manager"]
        return object_manager.get_static_object_array(vehicle) # (5, 10)

    def _get_neighbors(self, vehicle: BaseVehicle) -> np.ndarray:
        traffic_manager = vehicle.engine.managers["traffic_manager"]
        return traffic_manager.get_neighbors_history(vehicle) # (5, 10)


    def observe(self,
                vehicle: BaseVehicle = None,
                *args,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        vehicle와 vehicle.engine.current_map.road_network(= NodeRoadNetwork)을 통해
        extract_local_lanes_in_square_bbox() 호출
        """
        lanes_array, nav_lanes_array = self._get_lanes(vehicle)
        static_objects = self._get_static_objects(vehicle)
        neighbors_history = self._get_neighbors(vehicle)
        # 결과 Dict으로 포장
        observation_dict = {
            "lanes_array": lanes_array.astype(np.float32),
            "nav_lanes_array": nav_lanes_array.astype(np.float32),
            "static_objects": static_objects.astype(np.float32),
            "neighbors_history": neighbors_history.astype(np.float32),
        }
        # observation_dict = self.observation_normalizer(observation_dict)
        self.current_observation = observation_dict
        return observation_dict


    def destroy(self):
        # 메모리 정리 등
        super().destroy()
        self.current_observation = None
