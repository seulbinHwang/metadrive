from stable_baselines3 import PPO
from metadrive.envs import MetaDriveEnv
import time


# 학습된 모델 불러오기
model = PPO.load("ppo_metadrive_model")

# 평가용 단일 환경 생성 (렌더링 활성화 가능)
eval_env = MetaDriveEnv({
    "num_scenarios": 10,    # 평가엔 가볍게 10개 정도의 맵만 사용
    "start_seed": 1000,     # 학습 때와 다른 시드 범위 (예: 1000~1009번 맵들)
    "traffic_density": 0.1,
    "use_render": True      # 3D 렌더링 창을 띄워서 주행 모습 시각화
})

if __name__ == '__main__':

    # 에이전트 평가 주행
    episodes = 5  # 예시: 5 에피소드 평가
    for ep in range(1, episodes+1):
        obs, info = eval_env.reset()        # 환경 리셋 (새 맵 시작)
        episode_reward = 0.0
        done = False
        print(f"=== 에피소드 {ep} 시작 ===")
        while not done:
            # 학습된 모델로부터 행동 예측 (deterministic=True로 결정론적 정책 사용)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            # (선택) 조금 딜레이를 주어 화면을 보기 편하게 함
            time.sleep(0.02)
            # 에피소드 종료 판단
            done = terminated or truncated
        print(f"에피소드 {ep} 종료 - 총 획득 보상: {episode_reward:.2f}")
    eval_env.close()
