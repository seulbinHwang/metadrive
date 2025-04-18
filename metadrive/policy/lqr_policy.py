from metadrive.policy.base_policy import BasePolicy
from diffusion_planner.planner.planner import outputs_to_trajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.state_representation import TimePoint
import gym
import numpy as np


class LQRPolicy(BasePolicy):
    """
    LQR policy for controlling the vehicle in the simulation environment.
    This policy uses a Linear Quadratic Regulator (LQR)
    to compute the control inputs based on the current state of the vehicle.
    """

    def __init__(self, control_object):
        super().__init__(control_object)
        self.dt = (
            self.control_object.engine.global_config["physics_world_step_size"]
            * self.control_object.engine.global_config["decision_repeat"])
        self._tracker = LQRTracker(
            discretization_time=self.dt,
            vehicle=control_object._nuplan_vehicle_params)

    def act(self, agent_id):
        future_trajectory = self.engine.external_actions[agent_id]
        ego_history = list(self.control_object.ego_history)
        # future_trajectory: np.ndarray (80, 4) # 80: number of future steps, 4: x, y, vx, vy
        # Convert future_trajectory to control inputs(acceleration and steering rate)
        trajectory = InterpolatedTrajectory(
            trajectory=outputs_to_trajectory(future_trajectory, ego_history))
        """
        self._ego_controller.update_state(iteration, next_iteration,
                                  ego_state, trajectory: InterpolatedTrajectory(AbstractTrajectory))
        =============
        sampling_time = next_iteration.time_point - current_iteration.time_point


        """
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
        self.action_info["action"] = action
        return action

    @classmethod
    def get_input_space(cls):
        _input_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        return _input_space
