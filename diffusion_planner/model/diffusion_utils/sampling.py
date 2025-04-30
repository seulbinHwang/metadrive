from typing import Dict
import torch
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


def dpm_sampler(
        model: torch.nn.Module,  # self.dit,
        x_T,  # # [B, P, (1 + V_future) * 4]
        other_model_params: Dict = {},  #####
        diffusion_steps=10,
        noise_schedule_params: Dict = {},
        dpm_solver_params: Dict = {},  #####
        model_wrapper_params: Dict = {},  #####
        sample_params: Dict = {}):
    """
other_model_params={
    "cross_c": ego_neighbor_encoding,  # [B, P-1, D]
    "route_encoding": route_encoding,  # [B, D]
    "neighbor_current_mask": neighbor_current_mask  # [B, P-1]
},
dpm_solver_params={
    "correcting_xt_fn": initial_state_constraint,
},
model_wrapper_params=
    {
    "classifier_fn":
        self._guidance_fn, # GuidanceWrapper
    "classifier_kwargs": {
        "model": self.dit,
        "model_condition": {
            "cross_c": ego_neighbor_encoding,
            "route_encoding": route_encoding,
            "neighbor_current_mask": neighbor_current_mask
        },
        "inputs": inputs,
        "observation_normalizer": self._observation_normalizer,
        "state_normalizer": self._state_normalizer
    },
    "guidance_scale":
        0.5,
    "guidance_type":
        "classifier"
        if self._guidance_fn is not None else "uncond"
},

    """
    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(schedule='linear',
                                             **noise_schedule_params)

        model_fn = dpm.model_wrapper(
            model,  # self.dit
            noise_schedule,
            model_type=model.model_type,  # or "x_start" or "v" or "score"
            model_kwargs=other_model_params,
            **model_wrapper_params)

        dpm_solver = dpm.DPM_Solver(
            model_fn,  # GuidanceWrapper
            noise_schedule,
            algorithm_type="dpmsolver++",
            **dpm_solver_params)  # w.o. dynamic thresholding

        # Steps in [10, 20] can generate quite good samples.
        # And steps = 20 can almost converge.
        sample_dpm = dpm_solver.sample(x_T,
                                       steps=diffusion_steps,
                                       order=2,
                                       skip_type="logSNR",
                                       method="multistep",
                                       denoise_to_zero=True,
                                       **sample_params)

    return sample_dpm
