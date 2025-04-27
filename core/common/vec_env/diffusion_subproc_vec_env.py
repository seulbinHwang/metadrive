from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union


class DiffusionSubprocVecEnv(SubprocVecEnv):
    """
    A subclass of SubprocVecEnv that adds support for Diffusion-based environments.
    """

    def env_method(self,
                   method_name: str,
                   *method_args,
                   indices: VecEnvIndices = None,
                   **method_kwargs) -> list[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        npc_actions = method_kwargs.get("npc_actions", None)
        if npc_actions is None:  # (B, P-1, V_future = 80, 4)
            raise ValueError("npc_actions must be provided in method_kwargs")
        assert isinstance(npc_actions, np.ndarray,
                          "npc_actions must be a numpy array")
        # npc_ids = method_kwargs.get("npc_ids", None)
        # if npc_ids is None:
        #     raise ValueError("npc_ids must be provided in method_kwargs")
        # assert isinstance(npc_ids, list, "npc_ids must be list")
        for remote, a_npc_actions in zip(target_remotes, npc_actions):
            a_method_kwargs = {
                "npc_actions": a_npc_actions,  # (P-1, V_future = 80, 4)
            }
            remote.send(
                ("env_method", (method_name, method_args, a_method_kwargs)))
        return [remote.recv() for remote in target_remotes]
