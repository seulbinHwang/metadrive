from stable_baselines3.common.policies import BasePolicy
########
import os
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, TypeVar, Union, Dict

import numpy as np
from gymnasium import spaces
from torch import nn
from core.common.torch_layers import DiffusionExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from diffusion_planner.model.module.decoder import Decoder
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from diffusion_planner.utils.config import Config as DiffusionPlannerConfig
from pathlib import Path

args_path = Path(
    '~/PycharmProjects/metadrive/checkpoints/args.json').expanduser()

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")

import torch
import torch.nn as nn


class TransformerCritic(nn.Module):
    """
    inputs
      - seq_encoding : Tensor (B, T, D)      ← encoder_outputs
      - route_encoding : Tensor (B, D)       ← route_encoding
    output
      - value : Tensor (B,) 또는 (B,1)        ← V(s)
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        add_pos_emb: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ------- ① 토큰 준비 --------------------------------------------------
        #   • value_token : 값 추정용 가상 토큰 (learnable)
        self.value_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.value_token, std=0.02)

        # 선택적 위치 임베딩 (T 길이만큼 learnable하거나 sinusoidal)
        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(
                1, 1024, hidden_dim))  # 1024 = 최대 T 길이(넉넉히)
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ------- ② Transformer 인코더 ----------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # LayerNorm → Attention 순서
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ------- ③ Head -------------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1, bias=False)  # V(s) 스칼라
        )

    # ------------------------------------------------------------------------
    def forward(self, seq_encoding: torch.Tensor,
                route_encoding: torch.Tensor) -> torch.Tensor:
        """
        seq_encoding : (B, T, D)
        route_encoding : (B, D)
        return        : (B, 1)
        """
        B, T, D = seq_encoding.shape
        assert D == self.hidden_dim, "hidden_dim mismatch"

        # (1) value 토큰, route 토큰 붙이기
        value_tok = self.value_token.expand(B, -1, -1)  # (B, 1, D)
        route_tok = route_encoding.unsqueeze(1)  # (B, 1, D)

        x = torch.cat([value_tok, seq_encoding, route_tok],
                      dim=1)  # (B, 1+T+1, D)

        # (2) 위치 임베딩 추가
        if self.add_pos_emb:
            x = x + self.pos_emb[:, :x.size(1)]

        # (3) Transformer 인코딩
        x = self.encoder(x)  # (B, 1+T+1, D)

        # (4) 맨 앞 value_token 의 표현을 MLP로 -> 스칼라 V(s)
        value = self.head(x[:, 0])  # (B, 1)
        return value  # .squeeze(-1)  # (B,) 로 쓸 수도 있어 편의상 squeeze


class DiffusionActorCriticPolicy(BasePolicy):
    """
    Diffusion Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO, DPPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer

    ### 이외에는, PPO 인스턴스 생성할 때, policy_kwargs: Dict[str, Any] 인자에
    넣어주면 됨.
    """

    def __init__(
        self,
        observation_space: spaces.Space,  ###
        action_space: spaces.Space,  ###
        lr_schedule: Schedule,  ###
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = False,
        use_sde: bool = False,  ###
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[
            BaseFeaturesExtractor] = DiffusionExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = False,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        fine_tuning: bool = False,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.diffusion_planner_config = DiffusionPlannerConfig(
            args_file=str(args_path))
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        # TODO: features_extractor_kwargs 에 , features_dim 을 넣어야 하는지 고민
        features_extractor_kwargs["config"] = self.diffusion_planner_config
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        # Default network architecture, from stable-baselines
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        (self.observation_normalizer
        ) = self.diffusion_planner_config.observation_normalizer
        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        self.use_sde = use_sde
        self._build(lr_schedule)
        if not fine_tuning:
            (self.enc_dict, self.dec_dict,
             self.route_dict) = self._load_state_dict()
            self._set_state_dict()

    def _set_state_dict(self):
        """
        state_dict 를 encoder, decoder, route_encoder 로 나누어 저장합니다.
        """
        # encoder_state_dict
        self.features_extractor: DiffusionExtractor
        self.features_extractor.encoder.load_state_dict(self.enc_dict,
                                                        strict=True)
        # decoder_state_dict
        self.diffusion_transformer.dit.load_state_dict(self.dec_dict,
                                                       strict=True)
        # route_encoder_state_dict
        self.features_extractor.route_encoder.load_state_dict(self.route_dict,
                                                              strict=True)
        if not self.share_features_extractor:
            self.vf_features_extractor.encoder.load_state_dict(self.enc_dict,
                                                               strict=True)
            self.vf_features_extractor.route_encoder.load_state_dict(
                self.route_dict, strict=True)

    def _load_state_dict(self) -> tuple[dict, dict, dict]:
        pth_path = os.path.join("checkpoints", "model.pth")
        raw = torch.load(pth_path, map_location="cpu")
        state_dict = raw['ema_state_dict']
        # DDP로 저장된 키 제거
        model_state_dict = {
            k[len("module."):]: v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }

        # 분할 호출
        (enc_dict, dec_dict,
         route_dict) = self._extract_state_dicts(model_state_dict)
        return enc_dict, dec_dict, route_dict

    @staticmethod
    def _extract_state_dicts(state_dict: dict) -> tuple[dict, dict, dict]:
        """
        state_dict에서 세 부분을 추출하고, 각 키에서 prefix를 제거합니다.

        Args:
            state_dict (dict): 전체 model state_dict

        Returns:
            encoder_dict (dict): 'encoder.' prefix가 제거된 키로 구성된 사전
            decoder_dict (dict): 'decoder.decoder.dit.' prefix가 제거된 키로 구성되며, 'route_encoder.'로 시작하는 항목은 제외
            route_encoder_dict (dict): 'decoder.decoder.dit.route_encoder.' prefix가 제거된 키로 구성된 사전
        """
        enc_prefix = "encoder."
        dec_prefix = "decoder.decoder.dit."
        route_prefix = dec_prefix + "route_encoder."

        encoder_dict = {
            k[len(enc_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(enc_prefix)
        }
        decoder_dict = {
            k[len(dec_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(dec_prefix) and not k.startswith(route_prefix)
        }
        route_encoder_dict = {
            k[len(route_prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(route_prefix)
        }
        return encoder_dict, decoder_dict, route_encoder_dict

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate

        - diffusion_transformer 을 만들어야 함
        - critic_net 을 만들어야 함
        - value_net 은 그대로 사용하면 됨

        """
        self.diffusion_transformer = Decoder(self.diffusion_planner_config)
        self.critic_net = TransformerCritic(
            hidden_dim=self.diffusion_planner_config.hidden_dim)
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs)

    def _predict(self,
                 observation: PyTorchObs,
                 deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        observation = self.observation_normalizer(observation)
        """
        features: Dict[str, torch.Tensor]
            "encoding"
            "route_encoding"
        """
        features = self.extract_features(observation,
                                         self.pi_features_extractor)
        decoder_outputs: Dict[str, torch.Tensor] = self.diffusion_transformer(
            features, observation)

        predictions = decoder_outputs['prediction']  # (B, P, V_future, 4)
        ego_predictions = predictions[:, 0].detach()  # (B, V_future = 80, 4)
        return ego_predictions

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # log_likelihood shape: (B)
        # TODO: log_likelihood 구하기? ( https://velog.io/@ad_official/DPM-Solver-에서-log-likelihood-구하기 )
        log_likelihood = torch.ones(obs.shape[0]).to(obs.device)
        obs = self.observation_normalizer(obs)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            features: Dict[str, torch.Tensor]
            # decoder_outputs: {"prediction": (B, P, V_future, 4)
            decoder_outputs: Dict[str,
                                  torch.Tensor] = self.diffusion_transformer(
                                      features, obs)
            predictions = decoder_outputs['prediction']  # (B, P, V_future, 4)
            ego_predictions = predictions[:, 0].detach().cpu().numpy().astype(
                np.float64)  # (B, 80, 4)
            values = self.critic_net(features['encoding'],
                                     features['route_encoding'])  # shape (B)
        else:
            pi_features, vf_features = features
            decoder_outputs: Dict[str,
                                  torch.Tensor] = self.diffusion_transformer(
                                      pi_features, obs)
            predictions = decoder_outputs['prediction']
            ego_predictions = predictions[:, 0].detach().cpu().numpy().astype(
                np.float64)
            values = self.critic_net(vf_features['encoding'],
                                     vf_features['route_encoding'])
        """
        ego_predictions: (B, 80, 4)
        values: (B, 1)
        log_likelihood: (B)
        """
        # 특히 log_likelihood 의 shape 확인하기
        return ego_predictions, values, log_likelihood

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        obs = self.observation_normalizer(obs)
        features = self.extract_features(obs, self.vf_features_extractor)
        values = self.critic_net(features['encoding'],
                                 features['route_encoding'])  # shape (B)
        return values

    def _extract_features(
            self, obs: PyTorchObs, features_extractor: BaseFeaturesExtractor
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use.
        :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Union[Dict[str, torch.Tensor], tuple[Dict[str, torch.Tensor], Dict[
            str, torch.Tensor]]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return self._extract_features(
                obs, self.features_extractor
                if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = self._extract_features(obs,
                                                 self.pi_features_extractor)
            vf_features = self._extract_features(obs,
                                                 self.vf_features_extractor)
            return pi_features, vf_features
