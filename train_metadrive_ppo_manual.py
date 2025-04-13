import os
from functools import partial

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor
""" Monitor
에피소드의 총 보상, 길이 등을 자동으로 기록
"""
import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from gym import spaces



def make_metadrive_env(use_monitor=True):
    """MetaDriveEnv 환경을 생성하고 필요시 Monitor로 래핑하여 반환"""
    config = {
        "num_scenarios": 1000,  # 사용할 랜덤 맵 수 (예: 1000개 맵)
        "start_seed": 0,  # 시드 시작값 (0번부터 순차 생성)
        "traffic_density": 0.1,  # 교통량 밀도 (기본값 0.1)
        # "discrete_action": False,  # 연속 행동 사용 (False가 기본값)
        # 추가 필요 설정이 있다면 이곳에 작성
        "traffic_mode": "trigger",
        "random_traffic": True,
    }
    env = MetaDriveEnv(config)
    if use_monitor:
        env = Monitor(env)  # 모니터 래핑으로 에피소드 보상 등의 로깅 지원
    return env


# 병렬 환경 개수 지정
if __name__ == '__main__':

    num_envs = 4  # 시스템 자원에 따라 4~8개 등으로 조절 가능

    # 재현성 있는 학습을 위해 난수 시드 설정
    seed = 42
    set_random_seed(seed)

    # SubprocVecEnv를 이용해 독립 프로세스에서 여러 환경 실행
    """ SubprocVecEnv
    병렬 프로세스로 여러 환경을 실행하기 위한 벡터화 래퍼
    """
    envs = SubprocVecEnv(
        [partial(make_metadrive_env, True) for _ in range(num_envs)])
    """ VecMonitor
    여러 환경 벡터를 모니터링하기 위한 래퍼
    각 환경의 Monitor로부터 로그를 수집하여 Stable-Baselines3의 전체 성능 통계를 제공
    """
    envs = VecMonitor(envs)  # 여러 env의 모니터링 정보를 통합
    """
    Box(low=0.0, high=1.0, shape=(259, ), dtype=np.float32)
    
        envs.observation_space: Box(-0.0, 1.0, (259,), float32)
        envs.action_space: Box(-1.0, 1.0, (2,), float32)
    """
    # PPO 모델 초기화: MLP 정책 네트워크 사용
    model = PPO(
        policy="MlpPolicy",
        env=envs,
        verbose=1,  # 학습 과정 콘솔 출력 (0:출력없음, 1:정보, 2:상세)
        tensorboard_log="./ppo_metadrive_tensorboard"  # TensorBoard 로그 디렉토리
    )
    """ total_timesteps
    학습할 총 타임스텝 수
    """
    # ----- 학습 초기화 -----
    total_timesteps = 100000  # 예시: 학습할 총 환경 스텝 수
    log_interval = 10  # 10회 업데이트마다 로그 출력
    callback = None  # 특별한 콜백 없음 (필요시 Stable-Baselines3 BaseCallback 사용 가능)

    # BaseAlgorithm._setup_learn() 호출: 환경, 버퍼, 로거 등 초기 세팅
    # reset_num_timesteps=True: 새로 학습 시작, tb_log_name="PPO"로 텐서보드 로그 그룹 지정
    total_timesteps, callback = model._setup_learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
        tb_log_name="PPO",
        progress_bar=False)

    # 학습 시작 콜백 호출 (예: 콜백이 있는 경우 설정 초기화)
    if callback is not None:
        callback.on_training_start(locals(), globals())

    # ----- 메인 학습 루프 -----
    iteration = 0  # 몇 번째 정책 업데이트인지를 세는 카운터
    # num_timesteps는 model.env.step() 호출 시 자동 증가하며, model._total_timesteps에 목표값이 있음
    while model.num_timesteps < total_timesteps:
        # 1. Rollout 수집: 현재 정책으로 n_steps 만큼 환경과 상호작용하여 rollout_buffer 채우기
        continue_training = model.collect_rollouts(
            env=model.env,
            callback=callback or model._dummy_callback,
            # 콜백이 None이면 내부에서 DummyCallback 사용
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=model.n_steps)
        # collect_rollouts가 False를 리턴하면 (예: 콜백이 학습 중지를 요청) 루프 탈출
        if not continue_training:
            break

        # 2. 수집 완료 -> 정책 업데이트 단계
        iteration += 1  # 이번에 수집한 rollout에 대해 업데이트 횟수 증가

        # 학습 진행률 갱신: 남은 progress 비율 계산 (학습률 스케줄 등에 사용)
        model._update_current_progress_remaining(model.num_timesteps,
                                                 total_timesteps)

        # (선택) log_interval마다 환경 관련 로그 출력
        if log_interval is not None and iteration % log_interval == 0:
            model.dump_logs(iteration)  # 최근 에피소드 보상/길이 및 시간 통계 로그 기록

        # 3. 정책 신경망 및 가치신경망 업데이트 (PPO.train 호출)
        model.train()  # 현재 rollout_buffer의 데이터로 정책/가치 함수 계산 그래디언트 업데이트

    # ----- 학습 종료 처리 -----
    if callback is not None:
        callback.on_training_end()  # 학습 종료 콜백 호출 (필요한 마무리 작업 수행)
