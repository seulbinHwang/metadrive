"""
PPO 에이전트 평가 스크립트 (단일/병렬 모두 지원)
"""

import time
from pathlib import Path
from functools import partial

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from metadrive.envs import MetaDriveEnv

# ─── 사용자 설정 ──────────────────────────────────────────
MODEL_PATH = Path(__file__).with_name("ppo_metadrive_model")  # .zip 생략
NUM_ENVS   = 1         # 1 ➜ 단일 / 2 이상 ➜ 병렬
N_EPISODES = 5         # 총 평가 에피소드 수
RENDER     = False     # 단일 평가에서만 True 권장
# ──────────────────────────────────────────────────────────


def make_env(rank: int):
    """SubprocVecEnv 가 요구하는 '환경 팩토리 함수' 생성자"""
    def _init():
        cfg = dict(
            num_scenarios=10,
            start_seed=1000 + rank,
            traffic_density=0.1,
            use_render=RENDER and NUM_ENVS == 1  # 병렬일 땐 렌더링 OFF
        )
        return MetaDriveEnv(cfg)
    return _init


def evaluate():
    """에이전트 평가 루프"""
    # ── 1. 벡터화 환경 만들기 ───────────────────────────
    if NUM_ENVS == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    env = VecMonitor(env)

    # ── 2. 모델 로드 ───────────────────────────────────
    model = PPO.load(MODEL_PATH)

    # ── 3. 평가 ────────────────────────────────────────
    obs = env.reset()
    ep_returns   = []
    cur_returns  = np.zeros(NUM_ENVS)

    while len(ep_returns) < N_EPISODES:
        """
        obs: (n, 19)
        """
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        cur_returns += reward

        # done 이 된 환경만 기록-저장
        for idx, d in enumerate(done):
            if d:
                ep_returns.append(cur_returns[idx])
                print(f"[env {idx}] episode return = {cur_returns[idx]:.2f}")
                cur_returns[idx] = 0.0

        if RENDER and NUM_ENVS == 1:
            time.sleep(0.02)

    print("\n=== evaluation finished ===")
    print("episode returns :", [f"{r:.2f}" for r in ep_returns])
    print("mean return     :", np.mean(ep_returns).round(2))
    env.close()


# ─── 메인 가드 필수!! ────────────────────────────────────
if __name__ == "__main__":
    evaluate()
