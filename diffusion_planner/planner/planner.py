import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
# from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
# from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
# from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
# from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
# from nuplan.planning.simulation.planner.abstract_planner import (
#     AbstractPlanner, PlannerInitialization, PlannerInput)
#
# from diffusion_planner.data_process.data_processor import DataProcessor
# from diffusion_planner.utils.config import Config


def outputs_to_trajectory(predictions: np.ndarray,
                          ego_state_history: Deque[EgoState],
                          future_horizon=8.,
                          step_interval=0.1) -> List[InterpolatableState]:
    heading = np.arctan2(predictions[:, 3], predictions[:, 2])[...,
                                                               None]  # T, 1
    predictions = np.concatenate([predictions[..., :2], heading],
                                 axis=-1)  # T, 3

    states = transform_predictions_to_states(
        predictions,  # [T, 3]
        ego_state_history,
        future_horizon,  # 8
        step_interval)  # 0.1

    return states
