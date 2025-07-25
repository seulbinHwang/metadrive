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
import colorsys

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


def convert_center_to_rear_axle(traj_center: np.ndarray, vehicle) -> np.ndarray:
    """
    차량 중심 기준 궤적을 뒷축 중심 기준 궤적으로 변환

    Args:
        traj_center: (T, 4) array of [x_center, y_center, cos_yaw, sin_yaw]
        vehicle: BaseVehicle 객체 (REAR_WHEELBASE 속성 사용)

    Returns:
        traj_rear_axle: (T, 4) array of [x_rear_axle, y_rear_axle, cos_yaw, sin_yaw]
    """
    # 차량의 REAR_WHEELBASE 사용 (중심에서 뒷축까지의 거리)
    rear_wheelbase = vehicle.REAR_WHEELBASE

    # 각 시점에서 차량의 방향 벡터 (뒤쪽 방향)
    cos_yaw = traj_center[:, 2]  # (T,)
    sin_yaw = traj_center[:, 3]  # (T,)

    # 뒷축 방향으로의 오프셋 벡터 계산 (차량 좌표계에서 뒤쪽은 -x 방향)
    offset_x = -rear_wheelbase * cos_yaw  # (T,)
    offset_y = -rear_wheelbase * sin_yaw  # (T,)

    # 뒷축 중심 좌표 계산
    x_rear_axle = traj_center[:, 0] + offset_x  # (T,)
    y_rear_axle = traj_center[:, 1] + offset_y  # (T,)

    # 결과 조합 (yaw는 그대로 유지)
    traj_rear_axle = np.column_stack(
        [x_rear_axle, y_rear_axle, cos_yaw, sin_yaw])

    return traj_rear_axle


def convert_multiple_npc_center_to_rear_axle(
        external_npc_actions: np.ndarray, traffic_vehicles: List) -> np.ndarray:
    """
    여러 NPC 차량의 중심 기준 궤적들을 뒷축 기준으로 변환

    Args:
        external_npc_actions: (N, T, 4) array where N is number of vehicles
        traffic_vehicles: List of traffic vehicle objects

    Returns:
        converted_actions: (N, T, 4) array with rear axle coordinates
    """
    converted_actions = np.zeros_like(external_npc_actions)

    for i, (npc_traj,
            vehicle) in enumerate(zip(external_npc_actions, traffic_vehicles)):
        # BaseVehicle의 REAR_WHEELBASE 사용
        converted_actions[i] = convert_center_to_rear_axle(npc_traj, vehicle)

    return converted_actions


def apply_center_to_rear_axle_conversion(
        external_npc_actions: np.ndarray, traffic_vehicles: List,
        valid_predicted_closest_idx: List) -> np.ndarray:
    """
    가장 가까운 차량들의 중심 좌표를 뒷축 좌표로 변환

    Args:
        external_npc_actions: (P, T, 4) array of predicted trajectories
        traffic_vehicles: List of all traffic vehicles
        valid_predicted_closest_idx: List of indices for closest vehicles

    Returns:
        external_npc_actions: Modified array with rear axle coordinates
    """
    if valid_predicted_closest_idx is not None and len(
            valid_predicted_closest_idx) > 0:
        # 가장 가까운 순서대로 정렬된 차량들
        valid_predicted_vehs = [
            traffic_vehicles[i] for i in valid_predicted_closest_idx
        ]

        # external_npc_actions의 차량 개수만큼만 변환 (P대)
        valid_predicted_num = len(valid_predicted_vehs)
        actions_to_convert = external_npc_actions[:valid_predicted_num]

        # 변환된 결과를 external_npc_actions에 다시 할당
        external_npc_actions[:
                             valid_predicted_num] = convert_multiple_npc_center_to_rear_axle(
                                 actions_to_convert, valid_predicted_vehs)

    return external_npc_actions


class DiffusionTrafficManager(HistoricalBufferTrafficManager):

    def __init__(self):
        super().__init__()
        # 시각화된 궤적 NodePath를 보관할 리스트
        self._traffic_traj_nodes: list = []
        dt = (self.engine.global_config["physics_world_step_size"] *
              self.engine.global_config["decision_repeat"])
        self._initial_idm_steps = int(
            2. / dt)  # 20초 동안 IDMPolicy 적용
        self.current_step = 0

    def reset(self):
        # 기존 reset 처리
        super().reset()
        # 궤적 그림 초기화
        self.current_step = 0
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
        external_npc = engine.external_npc_actions[:,
                                                   1:]  # [:, :1, :]  # (N, T, 4)
        # 각 traffic 차량의 글로벌 궤적 좌표 구하기
        # 3) 차량별로 한 궤적씩 변환 → world coords (T,2)
        for idx, npc_traj in enumerate(external_npc):  # npc_traj.shape == (T,4)
            # 만약 npc_traj 의 값이 전부 0이라면, skip
            if np.all(npc_traj == 0.):
                continue
            coords_g, yaws_g = ego_to_global(npc_traj, ego_pos, ego_yaw)
            # 4) 각 점을 월드에 짧은 선으로 찍기
            for idx in range(len(coords_g) - 1):
                x1, y1 = coords_g[idx]
                x2, y2 = coords_g[idx + 1]
                np_node = engine._draw_line_3d(
                    LVector3(x1, y1, 1.5),
                    LVector3(x2, y2, 1.5),
                    color=(0, 0, 1, 1),  # 파랑
                    thickness=1)
                np_node.setMaterialOff(True)  # 재질(=Material) 완전히 제거
                np_node.reparentTo(engine.render)
                self._traffic_traj_nodes.append(np_node)
            # for (x, y) in coords_g:
            #     np_node = engine._draw_line_3d(
            #         LVector3(x, y, 1.5),
            #         LVector3(x, y, 3.),
            #         color=(0, 1, 0, 1),  # 초록
            #         thickness=3)
            #     np_node.reparentTo(engine.render)
            #     self._traffic_traj_nodes.append(np_node)
        external_guided_npc_actions = engine.external_guided_npc_actions
        if external_guided_npc_actions is None:
            return
        external_guided_npc_actions = external_guided_npc_actions[:, 1:]
        for idx, a_guided_npc_actions in enumerate(external_guided_npc_actions):
            # 만약 npc_traj 의 값이 전부 0이라면, skip
            if np.all(a_guided_npc_actions == 0.):
                continue
            coords_g, yaws_g = ego_to_global(a_guided_npc_actions, ego_pos,
                                             ego_yaw)
            for idx in range(len(coords_g) - 1):
                x1, y1 = coords_g[idx]
                x2, y2 = coords_g[idx + 1]
                np_node = engine._draw_line_3d(
                    LVector3(x1, y1, 1.5),
                    LVector3(x2, y2, 1.5),
                    color=(0, 0, 0, 1),  # 검정
                    thickness=3)
                np_node.reparentTo(engine.render)
                self._traffic_traj_nodes.append(np_node)

            # # 4) 각 점을 월드에 짧은 선으로 찍기
            # for (x, y) in coords_g:
            #     np_node = engine._draw_line_3d(
            #         LVector3(x, y, 1.5),
            #         LVector3(x, y, 3.),
            #         color=(1, 0, 0, 1),  # 빨강
            #         thickness=3)
            #     np_node.setMaterialOff(True)  # 재질(=Material) 완전히 제거
            #     np_node.reparentTo(engine.render)
            #     self._traffic_traj_nodes.append(np_node)

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
        if self.current_step >= self._initial_idm_steps:
            self._draw_all_traffic_trajs()
        external_npc_actions = self.engine.external_npc_actions[:,
                                                                1:]  # (P, 80, 4)


        predicted_agent_num = external_npc_actions.shape[0]
        if self.current_step < self._initial_idm_steps:
            # 초기 20초 동안은 IDMPolicy 적용
            predicted_agent_num = 0

        # ── 먼저 가장 가까운 P대 차량 찾기 ──
        valid_predicted_closest_idx = self._update_control_policies(
            0)

        # 변환 전 데이터 백업 (시각화용)
        external_npc_actions_before = external_npc_actions.copy()

        # NPC 차량 중심 좌표를 뒷축 좌표로 변환
        # external_npc_actions의 순서와 가장 가까운 차량들의 순서를 맞춰서 변환
        external_npc_actions = apply_center_to_rear_axle_conversion(
            external_npc_actions, self._traffic_vehicles,
            valid_predicted_closest_idx)

        # 변환 전후 비교 시각화 (선택적으로 활성화)
        if False:
            if valid_predicted_closest_idx is not None and len(
                    valid_predicted_closest_idx) > 0:
                try:
                    # 시각화 저장만 실행
                    plot_path = visualize_center_to_rear_axle_conversion(
                        external_npc_actions_before,
                        external_npc_actions,
                        self._traffic_vehicles,
                        valid_predicted_closest_idx,
                        save_dir=getattr(self.engine, 'conversion_plot_dir',
                                         './conversion_plots'))
                except Exception as e:
                    print(
                        f"Warning: Failed to save conversion visualization: {e}"
                    )
                    raise RuntimeError("test)")

        # (2) Ego 정보 한 번만 꺼내두기
        ego = next(iter(self.engine.agent_manager.active_agents.values()))
        ego_pos = np.array(ego.position[:2], dtype=np.float32)
        ego_yaw = ego.heading_theta
        if valid_predicted_closest_idx is not None:
            sorted_traffic_vehicles = [
                self._traffic_vehicles[i] for i in valid_predicted_closest_idx
            ]
            for vehicle_idx, veh in enumerate(sorted_traffic_vehicles):
                pol = self.engine.get_policy(veh.id)
                assert isinstance(pol, (LQRPolicy))
                npc_traj_wrt_ego = external_npc_actions[vehicle_idx]
                # ego → vehicle 로컬로 일괄 변환
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
        #         index = np.where(valid_predicted_closest_idx == vehicle_idx)[0][0]
        #         future_trajectory = external_npc_actions[index] # (80, 4)
        #         veh.before_step(pol.act(veh.id, future_trajectory))
        self.current_step += 1
        return {}

    # ────────────────────────────────────────────────────────────────────────
    # 내부 : 현 시점 traffic 차량들에 대해 “가까운 11대” 재계산 → policy 교체
    # ────────────────────────────────────────────────────────────────────────
    def _update_control_policies(self, predicted_agent_num=10) -> np.ndarray:
        if not self._traffic_vehicles or predicted_agent_num == 0:  # ── (0) early-return
            return None

        # ── (1) 대표 ego 선정 ──────────────────────────────────────────────
        ego_list = list(self.engine.agent_manager.active_agents.values())
        if not ego_list:
            raise RuntimeError("No active agents found in the scene.")
        ego_pos = ego_list[0].position  # (x, y, z) 또는 (x, y)

        # ── (2) 벡터화 거리 계산 & 11대 선별 ───────────────────────────────
        #      ->  Python loop 대신 NumPy C-루틴: GIL 해제 + SIMD 가능
        veh_positions = np.asarray([v.position for v in self._traffic_vehicles],
                                   dtype=np.float32)  # (total_num_agent,2/3)
        dists = np.linalg.norm(veh_positions - ego_pos,
                               axis=1)  # (total_num_agent,)
        total_num_agent = len(dists)

        valid_predicted_num = min(predicted_agent_num, total_num_agent)
        valid_predicted_closest_idx = np.argsort(dists)[:valid_predicted_num]
        lqr_target_set = {
            self._traffic_vehicles[i] for i in valid_predicted_closest_idx
        }

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
        return valid_predicted_closest_idx

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random,
                                           p=[0.2, 0.3, 0.3, 0.2, 0.0],
                                           vehicle_type="bicycle_history")
        return vehicle_type


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os


def visualize_center_to_rear_axle_conversion(
        external_npc_actions_before: np.ndarray,
        external_npc_actions_after: np.ndarray,
        traffic_vehicles: List,
        valid_predicted_closest_idx: List,
        save_dir: str = "./conversion_plots") -> str:
    """
    중심 좌표 → 뒷축 좌표 변환 전후를 시각적으로 비교하여 파일로 저장

    Args:
        external_npc_actions_before: (P, T, 4) 변환 전 궤적
        external_npc_actions_after: (P, T, 4) 변환 후 궤적
        traffic_vehicles: List of traffic vehicle objects
        valid_predicted_closest_idx: List of indices for closest vehicles
        save_dir: 저장할 디렉토리 경로 (사용하지 않음)

    Returns:
        str: 저장된 파일 경로
    """
    # 현재 레포지토리의 가장 상위 경로에 test.png로 저장
    filepath = "test.png"

    num_vehicles = min(len(valid_predicted_closest_idx),
                       external_npc_actions_before.shape[0])

    # 서브플롯 생성 (차량별로 비교)
    fig, axes = plt.subplots(2, (num_vehicles + 1) // 2, figsize=(15, 10))
    if num_vehicles == 1:
        axes = np.array([axes]).flatten()
    elif num_vehicles <= 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(num_vehicles):
        ax = axes[i]

        # 변환 전 궤적 (파란색)
        traj_before = external_npc_actions_before[i]  # (T, 4)
        x_before = traj_before[:, 0]
        y_before = traj_before[:, 1]

        # 변환 후 궤적 (빨간색)
        traj_after = external_npc_actions_after[i]  # (T, 4)
        x_after = traj_after[:, 0]
        y_after = traj_after[:, 1]

        # 궤적 그리기
        ax.plot(x_before,
                y_before,
                'b-o',
                markersize=3,
                linewidth=2,
                label='Center (Before)',
                alpha=0.7)
        ax.plot(x_after,
                y_after,
                'r-s',
                markersize=3,
                linewidth=2,
                label='Rear Axle (After)',
                alpha=0.7)

        # 시작점 강조
        ax.plot(x_before[0],
                y_before[0],
                'bo',
                markersize=8,
                label='Start (Center)')
        ax.plot(x_after[0],
                y_after[0],
                'ro',
                markersize=8,
                label='Start (Rear Axle)')

        # 차량 정보 가져오기
        vehicle = traffic_vehicles[valid_predicted_closest_idx[i]]
        rear_wheelbase = vehicle.REAR_WHEELBASE
        vehicle_length = vehicle.LENGTH  # 실제 차량 길이
        vehicle_width = vehicle.WIDTH  # 실제 차량 너비

        # 첫 번째 점에서 차량 형태 그리기 (변환 전후 비교)
        if len(x_before) > 0 and len(x_after) > 0:
            # 차량 방향 (첫 번째 점)
            cos_yaw = traj_before[0, 2]
            sin_yaw = traj_before[0, 3]

            center_x, center_y = x_before[0], y_before[0]
            rear_x, rear_y = x_after[0], y_after[0]

            # 차량 사각형은 원래 위치(중심 기준)에 그리기 - 차량 자체는 안 움직임
            vehicle_rect = patches.Rectangle(
                (center_x - vehicle_length/2, center_y - vehicle_width/2),
                vehicle_length, vehicle_width,
                angle=np.degrees(np.arctan2(sin_yaw, cos_yaw)),
                linewidth=2, edgecolor='gray', facecolor='lightgray', alpha=0.3
            )
            ax.add_patch(vehicle_rect)

            # 뒷축 선분 그리기 (차량 너비만큼 가로지르는 선)
            # 뒷축은 차량 중심에서 REAR_WHEELBASE만큼 뒤쪽에 위치
            rear_axle_center_x = center_x - rear_wheelbase * cos_yaw
            rear_axle_center_y = center_y - rear_wheelbase * sin_yaw

            # 뒷축 선분의 양 끝점 계산 (차량 방향에 수직)
            axle_half_width = vehicle_width / 2
            perpendicular_x = -sin_yaw * axle_half_width  # 차량 방향에 수직
            perpendicular_y = cos_yaw * axle_half_width

            axle_left_x = rear_axle_center_x + perpendicular_x
            axle_left_y = rear_axle_center_y + perpendicular_y
            axle_right_x = rear_axle_center_x - perpendicular_x
            axle_right_y = rear_axle_center_y - perpendicular_y

            # 뒷축 선분 그리기 (점선)
            ax.plot([axle_left_x, axle_right_x], [axle_left_y, axle_right_y],
                   'r:', linewidth=2, alpha=0.8, label='Rear Axle')

            # # 중심점과 뒷축 연결선 그리기 - 궤적 변환을 보여줌
            # ax.plot([center_x, rear_x], [center_y, rear_y], 'k--', linewidth=1, alpha=0.5, label='Offset Vector')

        ax.set_title(
            f'Vehicle {i+1} ({vehicle.__class__.__name__})\n'
            f'L:{vehicle_length:.1f}m, W:{vehicle_width:.1f}m, REAR_WB:{rear_wheelbase:.3f}m'
        )
        ax.set_xlabel('X (ego coordinate)')
        ax.set_ylabel('Y (ego coordinate)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # 사용하지 않는 서브플롯 숨기기
    for j in range(num_vehicles, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Conversion visualization saved to: {filepath}")
    raise NotImplementedError("This function is not fully implemented yet.")
    return filepath
