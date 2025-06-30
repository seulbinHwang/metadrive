from core.diffusion_dppo.diffusion_ppo import DiffusionPPO
from metadrive.envs.diffusion_planner_env_w_npc import DiffusionPlannerEnv
from core.common.policies import DiffusionActorCriticPolicy
import os
import argparse
from datetime import datetime, timezone
import os

import wandb
RED = (1, 0, 0, 1)          # RGBA


def download_wandb_model(args):
    if args.entity is None or args.project is None:
        raise ValueError("entity / project 정보를 인자로 주거나 환경변수 WANDB_ENTITY, "
                         "WANDB_PROJECT에 설정해야 합니다.")

    api = wandb.Api()
    coll_name = f"{args.name}_best-model"
    alias_path = f"{args.entity}/{args.project}/{coll_name}:best"
    try:
        artifact = api.artifact(alias_path, type="model")
    except wandb.errors.CommError:
        print("❌  서버에 best 아티팩트가 아직 없습니다.")
        return None, False
    current_id  = artifact.id          # 고유 해시 :contentReference[oaicite:2]{index=2}
    current_ver = artifact.version     # 'v3' 같은 문자열 :contentReference[oaicite:3]{index=3}



    # 삭제할 파일 경로를 생성
    out_dir = os.path.join(args.out_dir, coll_name)
    out_pth_dir = os.path.join(out_dir, "best.pth")
    os.makedirs(out_dir, exist_ok=True)
    cache_file = os.path.join(out_dir, ".wandb_best_id")
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            prev_id = f.read().strip()
    else:
        prev_id = None
    # 3) 동일 버전이면 건너뜀
    if prev_id == current_id:
        print(f"✅  이미 최신(best) 버전 {current_ver} ({current_id}) 을 보유 중입니다.")
        return None, False

    # 4) 새 버전이면 다운로드
    print(f"⬇️  새 best 버전 {current_ver} ({current_id}) 감지 – 다운로드 시작")
    target_dir = artifact.download(root=out_dir)   # 기본동작: 같은 이름이면 덮어쓰기 :contentReference[oaicite:4]{index=4}

    # 5) 캐시 갱신
    with open(cache_file, "w") as f:
        f.write(current_id)
    # print
    created_str = artifact.created_at  # e.g. "2025-06-29T05:12:34.123456Z"
    # "Z" 제거하고 fromisoformat으로 파싱
    created_dt_utc = datetime.fromisoformat(created_str.rstrip("Z")).replace(
        tzinfo=timezone.utc)
    created_local = created_dt_utc.astimezone()  # 시스템 로컬 타임존으로 변환

    print("\n[완료] 다운로드 경로:", target_dir)
    print("  • 이름       :", artifact.name)
    print("  • 버전       :", artifact.version)
    print("  • 타입       :", artifact.type)
    print("  • 별칭       :", artifact.aliases)
    print("  • 설명       :", artifact.description or "—")
    print("  • 업로드 시각:", created_local.isoformat(sep=' ', timespec='seconds'))

    if artifact.metadata:
        print("  • 메타데이터:")
        for key, value in artifact.metadata.items():
            print(f"      - {key}: {value}")
    else:
        print("  • 메타데이터: 없음")
    return out_pth_dir, True
def parse_args():
    p = argparse.ArgumentParser(
        description="Download the most recent best‑model artifact.")
    # positional: {args.name}__{time}
    p.add_argument("--name",
                   default="test",
                   help="{args.name}"
                   )
    # optional: entity / project (환경변수 fallback)
    p.add_argument("--entity",
                   "-e",
                   default=os.getenv("WANDB_ENTITY"),
                   help="W&B entity (team or user)")
    p.add_argument("--project",
                   "-p",
                   default=os.getenv("WANDB_PROJECT"),
                   help="W&B project name")
    p.add_argument("--out_dir",
                   "-o",
                   default=".",
                   help="Directory to download the file(s) into")
    return p.parse_args()



def main():
    args = parse_args()
    out_pth_dir, get_model = download_wandb_model(args)
    if not get_model:
        raise FileNotFoundError(
            f"클라우드에서 '{args.name}' best‑model 아티팩트를 찾을 수 없습니다.")
    config = {
        "num_scenarios": 20,
        "start_seed": 105,
        "traffic_density": 0.4,
        "map": 15,
        "random_traffic": True,
        "use_render": True,  # <- 화면 창 활성화
        "debug": False,
        "accident_prob": 0.,
    }
    policy_kwargs = {"pth_path": out_pth_dir}
    env = DiffusionPlannerEnv(config)
    model = DiffusionPPO(
        policy=DiffusionActorCriticPolicy,
        env=env,
        verbose=1,  # 학습 과정 콘솔 출력 (0:출력없음, 1:정보, 2:상세)
        tensorboard_log="./ppo_metadrive_tensorboard",  # TensorBoard 로그 디렉토리
        policy_kwargs=policy_kwargs
    )

    N_EPISODES = 5
    obs, _ = env.reset()
    episode_num = 0
    while episode_num < N_EPISODES:
        """
        obs: (n, 19)
        """
        action, _ = model.predict(obs, deterministic=True)
        npc_predictions, guided_npc_predictions = model.get_npc_predictions(obs) # ( P-1, V_future = 80, 4)
        env.set_external_npc_actions(npc_predictions, guided_npc_predictions)
        """
        만약 VecEnv 였으면,
        env.env_method(method_name="set_external_npc_actions",
                           npc_actions=npc_predictions)
        """
        obs, reward, terminated, truncated, info = env.step(action)
        # done 이 된 환경만 기록-저장
        if terminated or truncated:
            episode_num += 1
            env.reset()
    env.close()


if __name__ == "__main__":
    main()
