import os
from functools import partial

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor
""" Monitor
에피소드의 총 보상, 길이 등을 자동으로 기록
"""


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
    """
key: overtake_vehicle_num type(value): <class 'int'>
value: 0
key: velocity type(value): <class 'float'>
value: 0.0
key: steering type(value): <class 'float'>
value: 0.0
key: acceleration type(value): <class 'float'>
value: 0.0
key: step_energy type(value): <class 'float'>
value: 0.0
key: episode_energy type(value): <class 'float'>
value: 0.0
key: policy type(value): <class 'str'>
value: EnvInputPolicy
key: navigation_command type(value): <class 'str'>
value: forward
key: navigation_forward type(value): <class 'bool'>
value: True
key: navigation_left type(value): <class 'bool'>
value: False
key: navigation_right type(value): <class 'bool'>
value: False
key: crash_vehicle type(value): <class 'bool'>
value: False
key: crash_object type(value): <class 'bool'>
value: False
key: crash_building type(value): <class 'bool'>
value: False
key: crash_human type(value): <class 'bool'>
value: False
key: crash_sidewalk type(value): <class 'bool'>
value: False
key: out_of_road type(value): <class 'bool'>
value: False
key: arrive_dest type(value): <class 'bool'>
value: False
key: max_step type(value): <class 'bool'>
value: False
key: env_seed type(value): <class 'int'>
value: 369
key: crash type(value): <class 'bool'>
value: False
key: step_reward type(value): <class 'float'>
value: 0.0
key: route_completion type(value): <class 'numpy.float64'>
value: 0.011873212052783888
key: cost type(value): <class 'int'>
value: 0

    """
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
    total_timesteps = 500_000

    # PPO 알고리즘 학습 실행
    """ log_interval
    - 로그 출력 주기로, 10이면 10번의 모델 업데이트마다 콘솔에 중간 결과를 출력합니다.
    - 병렬 환경 4개에서 n_steps 기본값이 2048일 경우, 약 2048*4*10 = 81920 스텝마다 로그가 찍히는 셈
    """
    model.learn(total_timesteps=total_timesteps, log_interval=10)
    # 학습 완료 후 모델 저장
    model.save("ppo_metadrive_model")
    print("Training finished!")
    envs.close()
