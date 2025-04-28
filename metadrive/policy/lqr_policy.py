from metadrive.policy.base_policy import BasePolicy
from diffusion_planner.planner.planner import outputs_to_trajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.state_representation import TimePoint
import gymnasium as gym
import numpy as np
from panda3d.core import LVector3
from metadrive.constants import RENDER_MODE_NONE


class LQRPolicy(BasePolicy):
    """
    LQR policy for controlling the vehicle in the simulation environment.
    This policy uses a Linear Quadratic Regulator (LQR)
    to compute the control inputs based on the current state of the vehicle.
    """

    def __init__(self, control_object, random_seed=None, config=None):
        super().__init__(control_object, random_seed, config)
        self.dt = (
            self.control_object.engine.global_config["physics_world_step_size"]
            * self.control_object.engine.global_config["decision_repeat"])
        self._tracker = LQRTracker(
            discretization_time=self.dt,
            vehicle=control_object._nuplan_vehicle_params)
        # ───────── 시각화용 캐시 ─────────
        self._hist_np_list: list = []  # ← 새로 추가
        self._traj_np_list: list = []  # 직전 프레임에 그려 둔 NodePath들

    def reset(self):
        super().reset()
        for np_node in self._hist_np_list:
            np_node.removeNode()
        self._hist_np_list.clear()
        for np_node in self._traj_np_list:
            np_node.removeNode()
        self._traj_np_list.clear()

    def act(self, agent_id, future_trajectory=None):
        if future_trajectory is None:
            external_actions = self.engine.external_actions
            future_trajectory = external_actions[agent_id]
            return 0., 0.
        else:
            return 0., 0.
        ego_history = list(self.control_object.ego_history)
        # future_trajectory: np.ndarray (80, 4) # 80: number of future steps, 4: x, y, cos(yaw), sin(yaw)
        # Convert future_trajectory to control inputs(acceleration and steering rate)
        # List[EgoState]
        trajectory = InterpolatedTrajectory(
            trajectory=outputs_to_trajectory(future_trajectory, ego_history))
        # ② 궤적을 화면에 그리기
        # self._draw_history()
        # self._draw_trajectory(trajectory)
        # Compute the dynamic state to propagate the model
        ego_state = ego_history[-1]
        current_iteration = SimulationIteration(
            time_point=ego_state.time_point,
            index=self.control_object.engine.episode_step,
        )
        time_gap = TimePoint(int(self.dt * 1e6))
        next_iteration = SimulationIteration(
            time_point=ego_state.time_point + time_gap,
            index=self.control_object.engine.episode_step + 1,
        )
        action = self._tracker.track_trajectory(current_iteration,
                                                next_iteration, ego_state,
                                                trajectory)
        np.set_printoptions(suppress=True)  # 과학적 표기 억제
        # print("action", np.round(action, 2))
        self.action_info["action"] = action
        return action

    # ---------  여기가 ‘점 궤적 시각화’ 핵심 함수 ----------------------------
    def _draw_history(self) -> None:
        engine = self.control_object.engine
        if engine.mode == RENDER_MODE_NONE:
            return
        # 지난 프레임의 점들 제거
        for np_node in self._hist_np_list:
            np_node.removeNode()
        self._hist_np_list.clear()

        # ego_history는 오래된 → 최신 순으로 저장됨
        for ego_state in self.control_object.ego_history:
            # rear‑axle 기준 좌표를 사용
            x, y = ego_state.rear_axle.x, ego_state.rear_axle.y
            z = 1.4  # 살짝 위로 띄워서 노면과 구분
            np_dot = engine._draw_line_3d(
                LVector3(x, y, z),
                LVector3(x, y, z + 10.4),
                color=(1, 0, 0, 1),  # RED (원하는 색으로)
                thickness=140,  # 점/선 굵기
            )
            # ***월드에 부착***
            np_dot.reparentTo(engine.render)
            self._hist_np_list.append(np_dot)

    def _draw_trajectory(self, traj: InterpolatedTrajectory) -> None:
        """렌더링 창에 빨간 점(짧은 세로선)으로 궤적을 표시한다."""

        engine = self.control_object.engine
        if engine.mode == RENDER_MODE_NONE:  # 오프스크린/헤드리스일 때 무시
            return

        # 1) 지난 프레임 点 NodePath 정리
        for np_node in self._traj_np_list:
            np_node.removeNode()
        self._traj_np_list.clear()

        # 2) Trajectory 샘플 추출  (원하는 해상도로 Downsample 가능)
        sampled_states = traj.get_sampled_trajectory()  # List[EgoState]
        # 3) 각 점을 월드 좌표로 변환해 그린다
        for state in sampled_states:
            x, y, z = state.rear_axle.x, state.rear_axle.y, 1.5
            np_dot = engine._draw_line_3d(
                LVector3(x, y, z),
                LVector3(x, y, z + 12.52),  # 매우 짧은 선 = 점처럼 보임
                color=(0, 0, 1, 1),  # 빨간색
                thickness=160  # 점 크기 조절
            )
            # **중요!** 월드에 부착해야 보인다
            np_dot.reparentTo(engine.render)
            self._traj_np_list.append(np_dot)

    @classmethod
    def get_input_space(cls):
        """
        The planner passes an (80, 4) array:
            80  : prediction horizon steps
             4  : [x, y, cos(yaw), sin(yaw)]
        """
        return gym.spaces.Box(
            low=-1e10,
            high=1e10,
            shape=(80, 4),  # <-- 수정된 부분
            dtype=np.float32,
        )
