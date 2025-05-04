import copy
import logging
from collections import namedtuple
from typing import Dict, List, Optional
from panda3d.core import LVector3
import math
from metadrive.manager.traffic_manager import TrafficMode, HistoricalBufferTrafficManager
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative, TrackedObjectType, AgentInternalIndex, EgoInternalIndex
import numpy as np
from typing import List

from metadrive.policy.advanced_idm_policy import IDMPolicy
from metadrive.policy.lqr_policy import LQRPolicy  # ← 이미 구현돼 있다고 가정
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import RENDER_MODE_NONE

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")



# ── new_mixed_traffic_manager.py ───────────────────────────────────────────────


def rotation_matrix(theta: float) -> np.ndarray:
    """주어진 θ로부터 2×2 회전 행렬 반환."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def ego_to_global(traj_ego: np.ndarray, ego_pos: np.ndarray,
                  ego_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """
    traj_ego: (T,4) array of [x_ego, y_ego, cos_ego_yaw, sin_ego_yaw]
    ego_pos: (2,) global 위치
    ego_yaw: 스칼라 ego yaw
    returns:
      coords_global: (T,2) global x,y
      yaw_global:   (T,) global yaw
    """
    coords_ego = traj_ego[:, :2]  # (T,2)
    yaw_ego_frame = np.arctan2(traj_ego[:, 3], traj_ego[:, 2])  # (T,)
    R_e2g = rotation_matrix(ego_yaw)  # ego→global 회전
    coords_global = coords_ego.dot(R_e2g.T) + ego_pos  # (T,2)
    yaw_global = yaw_ego_frame + ego_yaw  # (T,)
    return coords_global, yaw_global


def global_to_local(coords_global: np.ndarray, yaw_global: np.ndarray,
                    veh_pos: np.ndarray, veh_yaw: float) -> np.ndarray:
    """
    coords_global: (T,2), yaw_global: (T,)
    veh_pos: (2,), veh_yaw: 스칼라
    returns:
      future_traj: (T,4) array of [x_local, y_local, cos_local_yaw, sin_local_yaw]
    """
    R_g2v = rotation_matrix(-veh_yaw)  # global→veh 회전
    delta = coords_global - veh_pos  # (T,2)
    coords_local = delta.dot(R_g2v.T)  # (T,2)
    yaw_local = yaw_global - veh_yaw  # (T,)

    cos_l = np.cos(yaw_local)[:, None]  # (T,1)
    sin_l = np.sin(yaw_local)[:, None]  # (T,1)
    return np.concatenate([coords_local, cos_l, sin_l], axis=1)


def transform_trajectory(npc_traj_wrt_ego: np.ndarray, ego_pos: np.ndarray,
                         ego_yaw: float, veh_pos: np.ndarray,
                         veh_yaw: float) -> np.ndarray:
    """
    ego계 기준 npc_traj_wrt_ego → 각 vehicle 로컬계 기준 (T,4) trajectory.
    """
    coords_g, yaw_g = ego_to_global(npc_traj_wrt_ego, ego_pos, ego_yaw)
    return global_to_local(coords_g, yaw_g, veh_pos, veh_yaw)


class DiffusionTrafficManager(HistoricalBufferTrafficManager):

    def __init__(self):
        super().__init__()
        # 시각화된 궤적 NodePath를 보관할 리스트
        self._traffic_traj_nodes: list = []

    def reset(self):
        # 기존 reset 처리
        super().reset()
        # 궤적 그림 초기화
        self._clear_traffic_trajs()
        # safety cap: 항상 최대 10대
        # TODO: remove
        # if len(self._traffic_vehicles) > 13:
        #     self._traffic_vehicles = self._traffic_vehicles[:13]

    def _clear_traffic_trajs(self):
        """이전 프레임에 그렸던 궤적 NodePath를 모두 제거."""
        for np_node in self._traffic_traj_nodes:
            np_node.removeNode()
        self._traffic_traj_nodes.clear()

    def _draw_all_traffic_trajs(self):
        """
        engine.external_npc_actions((N, T, 4): x,y,cos(yaw),sin(yaw))를
        ego→global 변환 후, 각 차량 위치 궤적을 월드에 그린다.
        """
        engine = self.engine
        # active ego 위치/방향
        ego = next(iter(engine.agent_manager.active_agents.values()))
        ego_pos = np.array([ego.rear_axle_xy[0], ego.rear_axle_xy[1]])
        ego_yaw = ego.heading_theta

        # 외부 NPC들이 예측해온 궤적
        external_npc = engine.external_npc_actions  # [:, :1, :]  # (N, T, 4)
        # 각 traffic 차량의 글로벌 궤적 좌표 구하기
        # 3) 차량별로 한 궤적씩 변환 → world coords (T,2)
        for idx, npc_traj in enumerate(external_npc):  # npc_traj.shape == (T,4)
            # 만약 npc_traj 의 값이 전부 0이라면, skip
            if np.all(npc_traj == 0.):
                continue
            coords_g, yaws_g = ego_to_global(npc_traj, ego_pos, ego_yaw)
            # 4) 각 점을 월드에 짧은 선으로 찍기
            for (x, y) in coords_g:
                np_node = engine._draw_line_3d(
                    LVector3(x, y, 1.5),
                    LVector3(x, y, 5.),
                    color=(0, 1, 0, 1),  # 초록
                    thickness=5)
                np_node.reparentTo(engine.render)
                self._traffic_traj_nodes.append(np_node)

    # ────────────────────────────────────────────────────────────────────────
    # reset 단계 – 트래픽 차량을 만든 뒤 1회 초기 분배
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # 매 step 직전 먼저 policy 구성을 갱신한 뒤, 부모 로직 수행
    # ────────────────────────────────────────────────────────────────────────
    def before_step(self):
        """
        1) block trigger → 새 traffic 차량을 self._traffic_vehicles 에 등록
        2) 가장 가까운 10 대에 LQRPolicy 부여
        3) 각 vehicle.before_step(action) 호출
        """
        self._clear_traffic_trajs()
        self._draw_all_traffic_trajs()
        external_npc_actions = self.engine.external_npc_actions[:,
                                                                1:]  # (P-1, 80, 4)
        diffusion_vehicle_num = external_npc_actions.shape[0]
        # ── 2.  이제 리스트가 확정됐으므로 policy 재배치
        closest_idx = self._update_control_policies(0)
        # (2) Ego 정보 한 번만 꺼내두기
        ego = next(iter(self.engine.agent_manager.active_agents.values()))
        ego_pos = np.array(ego.position[:2], dtype=np.float32)
        ego_yaw = ego.heading_theta
        if closest_idx is not None:
            sorted_traffic_vehicles = [
                self._traffic_vehicles[i] for i in closest_idx
            ]
            for vehicle_idx, veh in enumerate(sorted_traffic_vehicles):
                pol = self.engine.get_policy(veh.id)
                assert isinstance(pol, (LQRPolicy))
                npc_traj_wrt_ego = external_npc_actions[vehicle_idx]
                # global → vehicle 로컬로 일괄 변환
                future_traj = transform_trajectory(
                    npc_traj_wrt_ego, ego_pos, ego_yaw,
                    np.array(veh.position[:2], dtype=np.float32),
                    veh.heading_theta)
                veh.before_step(pol.act(veh.id, future_traj))

        # ── 1.  block trigger 처리 (부모 로직 그대로)
        if self.mode != TrafficMode.Respawn:
            for ego in self.engine.agent_manager.active_agents.values():
                if self.block_triggered_vehicles:
                    ego_road = Road(ego.lane_index[0], ego.lane_index[1])
                    if ego_road == self.block_triggered_vehicles[
                            -1].trigger_road:
                        blk = self.block_triggered_vehicles.pop()
                        self._traffic_vehicles += list(
                            self.get_objects(blk.vehicles).values())

        for vehicle_idx, veh in enumerate(self._traffic_vehicles):
            pol = self.engine.get_policy(veh.id)
            if isinstance(pol, IDMPolicy):
                veh.before_step(pol.act(is_kinematic=True))

        #
        # # ── 3.  action 적용
        # for vehicle_idx, veh in enumerate(self._traffic_vehicles):
        #     pol = self.engine.get_policy(veh.id)
        #     if isinstance(pol, IDMPolicy):
        #         veh.before_step(pol.act(is_kinematic=True))
        #     elif isinstance(pol, LQRPolicy):
        #         index = np.where(closest_idx == vehicle_idx)[0][0]
        #         future_trajectory = external_npc_actions[index] # (80, 4)
        #         veh.before_step(pol.act(veh.id, future_trajectory))
        return {}

    # ────────────────────────────────────────────────────────────────────────
    # 내부 : 현 시점 traffic 차량들에 대해 “가까운 11대” 재계산 → policy 교체
    # ────────────────────────────────────────────────────────────────────────
    def _update_control_policies(self, diffusion_vehicle_num=10) -> np.ndarray:
        if not self._traffic_vehicles or diffusion_vehicle_num == 0:  # ── (0) early-return
            return None

        # ── (1) 대표 ego 선정 ──────────────────────────────────────────────
        ego_list = list(self.engine.agent_manager.active_agents.values())
        if not ego_list:
            raise RuntimeError("No active agents found in the scene.")
        ego_pos = ego_list[0].position  # (x, y, z) 또는 (x, y)

        # ── (2) 벡터화 거리 계산 & 11대 선별 ───────────────────────────────
        #      ->  Python loop 대신 NumPy C-루틴: GIL 해제 + SIMD 가능
        veh_positions = np.asarray([v.position for v in self._traffic_vehicles],
                                   dtype=np.float32)  # (N, 2/3)
        dists = np.linalg.norm(veh_positions - ego_pos, axis=1)  # (N,)

        diffusion_vehicle_num = min(diffusion_vehicle_num, len(dists))
        closest_idx = np.argsort(dists)[:diffusion_vehicle_num]
        # closest_idx = np.argpartition(
        #     dists, diffusion_vehicle_num)[:diffusion_vehicle_num]  # k 개 인덱스
        lqr_target_set = {self._traffic_vehicles[i] for i in closest_idx}

        # ── (3) 교체가 필요한 차량만 따로 모아 한 번에 처리 ────────────────
        swap_cache = []  # (veh, desired_cls)
        get_policy = self.engine.get_policy
        for veh in self._traffic_vehicles:
            desired_cls = LQRPolicy if veh in lqr_target_set else IDMPolicy
            current_pol = get_policy(veh.id)
            if current_pol is None or not isinstance(current_pol, desired_cls):
                swap_cache.append((veh, desired_cls))

        # 실제 엔진 state 를 바꾸는 작업은 최소 loop 로
        for veh, cls in swap_cache:
            # engine.add_policy → BasePolicy(control_object, random_seed, …)
            self.add_policy(veh.id, cls, veh, self.generate_seed())
        return closest_idx

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random,
                                           p=[0.2, 0.3, 0.3, 0.2, 0.0],
                                           vehicle_type="bicycle_history")
        return vehicle_type
