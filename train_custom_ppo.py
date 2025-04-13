from stable_baselines3.common.vec_env import SubprocVecEnv
from metadrive.envs.custom_env import MultiModalMetaDriveEnv
from custom_extractor import MultiModalAttentionExtractor


def make_env(seed):

    def _init():
        env = MultiModalMetaDriveEnv(config={...
                                            })  # 필요에 따라 적절한 MetaDrive config 사용
        env.seed(seed)
        return env

    return _init


# 4개의 병렬 환경 생성
num_envs = 4
envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])

from stable_baselines3 import PPO

policy_kwargs = {
    'features_extractor_class':
        MultiModalAttentionExtractor,
    'features_extractor_kwargs':
        dict(embed_dim=128, n_attn_heads=4, n_layers=2),
    'net_arch': [],  # 추가적인 MLP 레이어 없음, extractor의 출력 그대로 사용
    # 'share_features_extractor': True,  # PPO의 기본값은 True이며, 액터와 크리틱이 extractor를 공유함
}
model = PPO(policy="MultiInputPolicy",
            env=envs,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./ppo_metadrive_tensorboard")

model.learn(total_timesteps=1_000_000)
model.save("ppo_metadrive_multimodal")
