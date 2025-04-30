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
from metadrive.constants import RENDER_MODE_ONSCREEN, BKG_COLOR, RENDER_MODE_NONE
from panda3d.core import LVector3, Vec4
from typing import Any, List, Tuple  # ← 맨 위 import 확인


def _to_vec4(color_tuple):
    """(r,g,b,a) -> Vec4; 값 범위 0~1 로 보정"""
    r, g, b, a = color_tuple
    return Vec4(float(r), float(g), float(b), float(a))


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


def extract_centerline_in_ego_frame(lane,
                                    ego_position_world: np.ndarray,
                                    ego_heading_world: float,
                                    step: float = 0.5,
                                    M_max: int = 200) -> np.ndarray:
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
        ego: BaseVehicle, lane_roi_length: float, max_lane_num: int,
        max_route_num: int,
        lane_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    반환값을
        ▸ lanes                : (max_lane_num, lane_len, 12)
        ▸ lanes_speed_limit    : (max_lane_num, 1)   [float32, m/s]
        ▸ lanes_has_speed_limit: (max_lane_num, 1)   [bool]
        ▸ route_lanes      : (max_route_num, lane_len, 12)
    로 확장합니다.
    """
    road_network = ego.engine.current_map.road_network
    checkpoint_node_ids = ego.navigation.checkpoints
    ego_x, ego_y = ego.rear_axle_xy
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
    lanes = np.zeros((max_lane_num, lane_len, 12), dtype=np.float32)
    lanes_speed_limit = np.zeros((max_lane_num, 1), dtype=np.float32)
    lanes_has_speed_limit = np.zeros((max_lane_num, 1), dtype=np.bool_)

    route_lanes = np.zeros((max_route_num, lane_len, 12), dtype=np.float32)
    nav_idx = 0
    for idx, (ln,
              ln_idx_tuple) in enumerate(zip(selected_lanes, selected_indices)):
        ln_len = ln.length
        sl = getattr(ln, "speed_limit", None)  # 존재하지 않으면 None
        # TODO: check
        # if sl is not None and sl > 0:
        #     lanes_speed_limit[idx, 0] = sl * (1000 / 3600
        #                                      )  # sl * (1000/3600) 로 m/s 변환
        #     lanes_has_speed_limit[idx, 0] = True

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
        lanes[idx, :, 0] = local_cx
        lanes[idx, :, 1] = local_cy
        lanes[idx, :, 2] = diffs[:, 0]
        lanes[idx, :, 3] = diffs[:, 1]

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

        lanes[idx, :, 4] = left_off[:, 0]
        lanes[idx, :, 5] = left_off[:, 1]
        lanes[idx, :, 6] = right_off[:, 0]
        lanes[idx, :, 7] = right_off[:, 1]
        # TODO: 신호등 환경에 만들고 수정 필요
        lanes[idx, :, 8] = 1.0  # [1,0,0,0]
        start_node_str = ln_idx_tuple[0]
        end_node_str = ln_idx_tuple[1]
        if nav_idx < max_route_num:
            if (start_node_str
                    in checkpoint_node_ids) and (end_node_str
                                                 in checkpoint_node_ids):
                route_lanes[nav_idx] = lanes[idx]
                nav_idx += 1
    return lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes


class DiffusionPlannerObservation(BaseObservation):
    """
    이 클래스는 extract_local_lanes_in_square_bbox()를 호출해,
    lanes_array와 route_lanes 두 가지 np.ndarray 정보를 Dict로 묶어 반환한다.
    """

    def __init__(self, config):
        super().__init__(config)
        self.observation_normalizer = config.observation_normalizer
        # 예: config 내부에 아래 key들이 있다고 가정
        self.lane_roi_length = self.config.get("lane_roi_length", 200.0)
        self.max_lane_num = self.config.get("max_lane_num", 70)
        self.max_route_num = self.config.get("max_route_num", 25)
        self.lane_len = self.config.get("lane_len", 20)
        self.max_obj = self.config.get("max_obj", 5)

        # lanes shape: (max_lane_num, lane_len, 12)
        # route_lanes shape: (max_route_num, lane_len, 12)
        # dtype=float32, 값 범위는 특정 제한 없으므로 Box(-inf, inf, ... ) etc.
        # 필요 시에는 관습적으로 아주 넓은 범위를 clip해도 됨
        self._observation_space = gym.spaces.Dict({
            "ego_current_state":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(10,),
                               dtype=np.float32),
            "lanes":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_lane_num, self.lane_len, 12),
                               dtype=np.float32),
            "lanes_speed_limit":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_lane_num, 1),
                               dtype=np.float32),
            "lanes_has_speed_limit":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_lane_num, 1),
                               dtype=np.float32),
            "route_lanes":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_route_num, self.lane_len, 12),
                               dtype=np.float32),
            "static_objects":
                gym.spaces.Box(low=-1e10,
                               high=1e10,
                               shape=(self.max_obj, 10),
                               dtype=np.float32),
            "neighbor_agents_past":
                gym.spaces.Box(
                    low=-1e10,
                    high=1e10,
                    shape=(32, 21, 11),  # TODO: hard coding 제거
                    dtype=np.float32),
        })
        # vis_mode = config.get("vis_mode", "all").lower()
        vis_mode = "route"
        assert vis_mode in {
            "none", "lanes", "route", "all", "neighbors", "static"
        }
        self._vis_lanes = vis_mode in {"lanes", "all"}
        self._vis_route = vis_mode in {"route", "all"}
        self._vis_neighbors = vis_mode in {"neighbors", "all"}  # ← 추가
        self._vis_static = vis_mode in {"static", "all"}  # ← NEW

        self._lane_np_list: list = []  # ← 여기에 그려둔 선 NodePath 보관
        self._route_np_list: list = []  # 경로 차선 선(NodePath) 캐시  ← NEW
        self._neighbor_np_list: List[Any] = []  # 주변 Agent NodePath
        self._static_np_list: List[Any] = []  # 정적 오브젝트 NodePath 캐시  ← NEW

    # ──────────────────────────────────────────────────────────
    #   정적 오브젝트(콘, 배리어, 워닝 트라이포드 …) 시각화
    #   static_array : (max_obj, 10)
    # ──────────────────────────────────────────────────────────
    def _visualize_static_objects(
        self,
        vehicle: "BaseVehicle",
        static_array: np.ndarray,
    ) -> None:
        engine: Any = vehicle.engine
        if engine.mode == RENDER_MODE_NONE or not self._vis_static:
            return

        # ── 지난 프레임 정리 ───────────────────────────────
        for np_node in self._static_np_list:
            np_node.removeNode()
        self._static_np_list.clear()

        ego_x, ego_y = vehicle.rear_axle_xy
        ego_yaw: float = float(vehicle.heading_theta)

        # 색상 팔레트: 0 Warning / 1 Barrier / 2 Cone / 3 기타
        palette: List[Tuple[float, float, float, float]] = [
            (1.0, 0.5, 0.0, 1.0),  # 주황 (Warning)
            (1.0, 0.0, 0.0, 1.0),  # 빨강  (Barrier)
            (1.0, 1.0, 0.0, 1.0),  # 노랑  (Cone)
            (0.6, 0.6, 0.6, 1.0),  # 회색  (기타)
        ]

        for obj in static_array:  # (10,)
            if np.allclose(obj, 0.0, atol=1e-6):
                continue

            lx, ly, c_h, s_h, width, length = obj[:6]
            one_hot = obj[6:10]
            type_idx: int = int(np.argmax(one_hot))
            col: Tuple[float, float, float, float] = palette[type_idx]

            # 로컬 → 월드 변환
            cx_w_arr, cy_w_arr = self._local_to_world_batch(
                np.array([lx]), np.array([ly]), ego_x, ego_y, ego_yaw)
            cx_w: float = float(cx_w_arr[0])
            cy_w: float = float(cy_w_arr[0])

            yaw_local: float = math.atan2(float(s_h), float(c_h))
            yaw_world: float = yaw_local + ego_yaw

            # 사각형(테두리) 그리기
            self._static_np_list += self._draw_box(
                engine,
                cx_w=cx_w,
                cy_w=cy_w,
                yaw=yaw_world,
                length=float(length),
                width=float(width),
                color=col,
            )

    @staticmethod
    def _local_vec_to_world(vx: float, vy: float,
                            ego_yaw: float) -> Tuple[float, float]:
        """(vx, vy) 를 병진 없이 yaw 만큼 회전한 월드벡터 반환"""
        c: float = math.cos(ego_yaw)
        s: float = math.sin(ego_yaw)
        wx: float = vx * c - vy * s
        wy: float = vx * s + vy * c
        return wx, wy

    # -----------------------------------------------
    # 3)  바운딩-박스(차량) 그리기
    # -----------------------------------------------
    def _draw_box(
        self,
        engine: Any,
        cx_w: float,
        cy_w: float,
        yaw: float,
        length: float,
        width: float,
        color: Tuple[float, float, float, float],
    ) -> List[Any]:
        """월드 좌표 중심/크기로 사각형 전체를 폴리라인으로 그리고, 중앙에서 heading 화살표를 그립니다."""
        hl: float = 0.5 * length
        hw: float = 0.5 * width
        c, s = math.cos(yaw), math.sin(yaw)

        corners: List[Tuple[float, float]] = []
        for dx, dy in [(+hl, +hw), (+hl, -hw), (-hl, -hw), (-hl, +hw),
                       (+hl, +hw)]:
            x: float = cx_w + dx * c - dy * s
            y: float = cy_w + dx * s + dy * c
            corners.append((x, y))

        # 1) 박스 테두리 그리기
        np_list = self._draw_polyline(
            engine=engine,
            pts_world=corners,
            color=color,
            dotted=False,
            thickness=60,
        )

        # 2) 중앙에서 heading 방향으로 화살표(선) 그리기
        #    길이를 박스 길이의 1.5배로
        arrow_length = length * 1.5
        # 박스 앞쪽 꼭짓점이 아닌 박스 중심에서 시작
        start = LVector3(cx_w, cy_w, 0.1)
        end_x = cx_w + arrow_length * c
        end_y = cy_w + arrow_length * s
        end = LVector3(end_x, end_y, 0.1)
        heading_node = engine._draw_line_3d(
            start,
            end,
            color=(1.0, 0.0, 0.0, 1.0),  # 진한 빨강으로
            thickness=80,
        )
        heading_node.reparentTo(engine.render)
        np_list.append(heading_node)

        return np_list

    # ──────────────── ② 메인 visualize 함수 ────────────────
    @staticmethod
    def _local_to_world_batch(px, py, ego_x, ego_y, ego_yaw):
        """벡터화된 로컬→월드 변환"""
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        wx = px * c - py * s + ego_x
        wy = px * s + py * c + ego_y
        return wx, wy

    def _visualize_neighbors(
            self,
            vehicle: "BaseVehicle",
            neighbor_array: np.ndarray,  # (32, 21, 11)
    ) -> None:
        engine: Any = vehicle.engine
        if engine.mode == RENDER_MODE_NONE or not self._vis_neighbors:
            return

        # ── 지난 프레임 정리 ────────────────────────────────
        for np_node in self._neighbor_np_list:
            np_node.removeNode()
        self._neighbor_np_list.clear()

        ego_x, ego_y = vehicle.rear_axle_xy
        ego_yaw: float = float(vehicle.heading_theta)

        vmax: float = 100.0 / 3.6  # 100 km/h  → m/s
        arrow_scale: float = 8.0  # 최대 화살표 길이 [m]
        # ── 각 주변 에이전트 루프 ──────────────────────────
        for agent_hist in neighbor_array:  # (21, 11)
            if np.allclose(agent_hist, 0.0, atol=1e-6):
                continue

            # (a) 색상 결정 -------------------------------------------------
            onehot: np.ndarray = agent_hist[-1, 8:11]
            if onehot[0] == 1:  # Vehicle
                col: Tuple[float, float, float, float] = (0.00, 0.60, 1.00, 1.0)
            elif onehot[2] == 1:  # Bicycle
                col = (1.00, 0.70, 0.00, 1.0)
            else:  # Pedestrian
                col = (0.00, 1.00, 0.00, 1.0)

            # (b) 과거 21 스텝 모두 시각화 -------------------------------
            #     오래된 샘플일수록 투명·얇게 해서 겹침을 줄임
            for t_idx in range(agent_hist.shape[0]):  # 0 (과거) → 20 (현재)
                x, y, c_h, s_h, *_tail = agent_hist[t_idx, :]
                width, length = _tail[2], _tail[3]  # w, l 순서 주의

                yaw_local: float = math.atan2(s_h, c_h)
                yaw_world: float = yaw_local + ego_yaw

                cx_w_arr, cy_w_arr = self._local_to_world_batch(
                    np.array([x]), np.array([y]), ego_x, ego_y, ego_yaw)
                cx_w: float = float(cx_w_arr[0])
                cy_w: float = float(cy_w_arr[0])

                # 투명도/두께: 가장 최근(20) = 굵고 불투명, 가장 과거(0) = 얇고 투명
                alpha: float = 0.15 + 0.85 * (t_idx / 20.0)
                thick: int = int(30 + 30 * (t_idx / 20.0))

                self._neighbor_np_list += self._draw_box(
                    engine,
                    cx_w,
                    cy_w,
                    yaw_world,
                    float(length),
                    float(width),
                    color=(col[0], col[1], col[2], alpha),
                )

            # (c) 현재 스텝(마지막 인덱스)만 속도 화살표 추가 --------------
            x, y, c_h, s_h, vx, vy, width, length = agent_hist[-1, :8]
            speed: float = math.hypot(vx, vy)
            if speed < 1e-2:
                continue
            vx_w, vy_w = self._local_vec_to_world(float(vx), float(vy), ego_yaw)
            norm: float = min(speed / vmax, 1.0)
            arr_len: float = norm * arrow_scale  # [m]

            cx_w_arr, cy_w_arr = self._local_to_world_batch(
                np.array([x]), np.array([y]), ego_x, ego_y, ego_yaw)
            cx_w, cy_w = float(cx_w_arr[0]), float(cy_w_arr[0])

            ux: float = vx_w / speed * arr_len
            uy: float = vy_w / speed * arr_len

            self._neighbor_np_list += self._draw_polyline(
                engine,
                [(cx_w, cy_w), (cx_w + ux, cy_w + uy)],
                color=col,
                dotted=False,
                thickness=40,
            )

    # -----------------------------------------------------------
    # ② 폴리라인(연속 선분) 그리기
    # -----------------------------------------------------------

    def _draw_polyline(self, engine, pts_world, color, dotted, thickness):
        """
        pts_world : [(x, y), ...]  world 좌표
        color     : (r, g, b) or (r, g, b, a)  ― 0~1 실수
        """
        # ① RGBA 정규화 --------------------------------------------------
        if len(color) == 3:
            r, g, b = color
            a = 1.0
        else:
            r, g, b, a = color

        np_list = []
        step = 2 if dotted else 1

        for p0, p1 in zip(pts_world[:-1:step], pts_world[1::step]):
            np_node = engine._draw_line_3d(
                LVector3(p0[0], p0[1], 0.35),
                LVector3(p1[0], p1[1], 0.35),
                color=(r, g, b),  # ← 원 함수 파라미터 (조명 필요)
                thickness=thickness,
            )

            # ★★ 핵심 3 줄 ― 조명·재질 끄고, 순수 VertexColor 사용 ★★
            np_node.setMaterialOff(True)  # 재질(=Material) 완전히 제거
            np_node.setLightOff(True)  # 조명 영향 제거
            np_node.setColor(r, g, b, a, 1)  # 우선순위 1 ⇒ 무조건 적용
            # ----------------------------------------------------------------

            np_node.reparentTo(engine.render)
            np_list.append(np_node)
        return np_list

    # -----------------------------------------------------------
    # ③ 일반 차선 + 경로 차선 시각화
    # -----------------------------------------------------------
    def _visualize_lanes(self, vehicle, lanes_array, route_array):
        engine = vehicle.engine
        if engine.mode == RENDER_MODE_NONE:
            return

        # ── 지난 프레임 선 모두 제거 ───────────────────────────
        for np_node in self._lane_np_list + self._route_np_list:
            np_node.removeNode()
        self._lane_np_list.clear()
        self._route_np_list.clear()

        # 아무것도 안 그리는 모드라면 여기서 끝
        if not (self._vis_lanes or self._vis_route):
            return

        # ── Ego 포즈 --------------------------------------------------
        ego_x, ego_y = vehicle.rear_axle_xy
        ego_yaw = vehicle.heading_theta

        # ── C. 일반 차선 --------------------------------------------
        if self._vis_lanes:
            for lane_i in lanes_array:
                if np.allclose(lane_i[:, :8], 0.0, atol=1e-6):
                    continue
                cx, cy = lane_i[:, 0], lane_i[:, 1]
                lx_off, ly_off = lane_i[:, 4], lane_i[:, 5]
                rx_off, ry_off = lane_i[:, 6], lane_i[:, 7]

                cx_w, cy_w = self._local_to_world_batch(cx, cy, ego_x, ego_y,
                                                        ego_yaw)
                lx_w, ly_w = self._local_to_world_batch(cx + lx_off,
                                                        cy + ly_off, ego_x,
                                                        ego_y, ego_yaw)
                rx_w, ry_w = self._local_to_world_batch(cx + rx_off,
                                                        cy + ry_off, ego_x,
                                                        ego_y, ego_yaw)

                self._lane_np_list += self._draw_polyline(
                    engine, list(zip(cx_w, cy_w)), (1, 1, 1, 1), True, 40)
                self._lane_np_list += self._draw_polyline(
                    engine, list(zip(lx_w, ly_w)), (1, 1, 0, 1), False, 50)
                self._lane_np_list += self._draw_polyline(
                    engine, list(zip(rx_w, ry_w)), (0.7, 0.7, 0.7, 1), False,
                    50)

        # ── D. 경로 차선 --------------------------------------------
        if self._vis_route:
            for route_i in route_array:
                if np.allclose(route_i, 0.0, atol=1e-6):
                    continue

                # 0~7 번째 칼럼은 일반 차선과 동일한 의미!
                cx, cy = route_i[:, 0], route_i[:, 1]  # 중앙선
                lx_off, ly_off = route_i[:, 4], route_i[:, 5]  # 왼쪽 offset
                rx_off, ry_off = route_i[:, 6], route_i[:, 7]  # 오른쪽 offset

                # 월드 좌표 변환
                cx_w, cy_w = self._local_to_world_batch(cx, cy, ego_x, ego_y,
                                                        ego_yaw)
                lx_w, ly_w = self._local_to_world_batch(cx + lx_off,
                                                        cy + ly_off, ego_x,
                                                        ego_y, ego_yaw)
                rx_w, ry_w = self._local_to_world_batch(cx + rx_off,
                                                        cy + ry_off, ego_x,
                                                        ego_y, ego_yaw)

                # ① 중앙선 ― 빨간 점선  (step=2 자동 적용)
                self._route_np_list += self._draw_polyline(
                    engine,
                    list(zip(cx_w, cy_w)),
                    color=(1, 0, 0, 1),  # 빨강
                    dotted=True,  # ← 점선
                    thickness=90,
                )
                # ② 왼쪽 경계 ― 빨간 실선
                self._route_np_list += self._draw_polyline(
                    engine,
                    list(zip(lx_w, ly_w)),
                    color=(1, 0, 0, 1),
                    dotted=False,  # 실선
                    thickness=90,
                )
                # ③ 오른쪽 경계 ― 빨간 실선
                self._route_np_list += self._draw_polyline(
                    engine,
                    list(zip(rx_w, ry_w)),
                    color=(1, 0, 0, 1),
                    dotted=False,
                    thickness=90,
                )

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    def _get_lanes(
        self, vehicle: BaseVehicle
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 실제 함수 호출
        (lanes, lanes_speed_limit, lanes_has_speed_limit,
         route_lanes) = extract_local_lanes_in_square_bbox(
             ego=vehicle,
             lane_roi_length=self.lane_roi_length,
             max_lane_num=self.max_lane_num,
             max_route_num=self.max_route_num,
             lane_len=self.lane_len)
        return lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes

    def _get_static_objects(self, vehicle: BaseVehicle) -> np.ndarray:
        object_manager = vehicle.engine.managers.get("object_manager")
        if object_manager is None:
            return np.zeros((self.max_obj, 10), dtype=np.float32)
        return object_manager.get_static_object_array(vehicle,
                                                      self.max_obj)  # (5, 10)

    def _get_neighbors(self, vehicle: BaseVehicle) -> np.ndarray:
        traffic_manager = vehicle.engine.managers["traffic_manager"]
        return traffic_manager.get_neighbors_history(vehicle)  # (32, 21, 10)

    def observe(self,
                vehicle: BaseVehicle = None,
                *args,
                **kwargs) -> Dict[str, np.ndarray]:
        """
        vehicle와 vehicle.engine.current_map.road_network(= NodeRoadNetwork)을 통해
        extract_local_lanes_in_square_bbox() 호출
        """
        (lanes, lanes_speed_limit, lanes_has_speed_limit,
         route_lanes) = self._get_lanes(vehicle)
        self._visualize_lanes(vehicle, lanes, route_lanes)
        static_objects = self._get_static_objects(vehicle)
        neighbor_agents_past = self._get_neighbors(vehicle)
        self._visualize_neighbors(vehicle, neighbor_agents_past)
        self._visualize_static_objects(vehicle, static_objects)  # ← NEW
        # 결과 Dict으로 포장
        observation_dict = {
            "ego_current_state":
                np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                         dtype=np.float32),
            "lanes":
                lanes.astype(np.float32),
            "lanes_speed_limit":
                lanes_speed_limit.astype(np.float32),
            "lanes_has_speed_limit":
                lanes_has_speed_limit.astype(np.float32),
            "route_lanes":
                route_lanes.astype(np.float32),
            "static_objects":
                static_objects.astype(np.float32),
            "neighbor_agents_past":
                neighbor_agents_past.astype(np.float32),
        }
        # observation_dict = self.observation_normalizer(observation_dict)
        self.current_observation = observation_dict
        return observation_dict

    def destroy(self):
        # 메모리 정리 등
        super().destroy()
        self.current_observation = None
