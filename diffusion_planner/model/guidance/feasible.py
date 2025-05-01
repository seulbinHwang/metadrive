import torch
import torch.nn.functional as F

# 물리적 최대 이동 거리 (delta_t 동안)
# inputs에서 r_max를 받아 사용합니다.
_EPS = 1e-6


def feasible_guidance_fn(
    x: torch.Tensor,
    t: torch.Tensor,
    cond,
    inputs: dict, *args, **kwargs
) -> torch.Tensor:
    """
    각 에이전트의 궤적이 물리적 이동 한계를 초과하지 않도록 평가하는 guidance 함수입니다.

    Args:
        x (torch.Tensor): [B*(Pn+1), T+1, 4] 형태의 trajectory tensor.
        t (torch.Tensor): [B, 1] 형태의 diffusion timestamp.
        cond: 조건 입력(사용하지 않음).
        inputs (dict): 추가 입력 dict로, 'r_max' key에 이동 한계 거리(scalar) 포함.

    Returns:
        torch.Tensor: [B] 형태의 guidance 값. 값이 클수록 제약을 위반했음을 나타냅니다.
    """
    # 배치 크기, 에이전트 수(Pn+1), timesteps
    B, Pp1, Tp1, _ = x.shape

    # reshape 및 gradient 적용 시점만 활성화
    x = x.reshape(B, Pp1, Tp1, 4)
    mask = (t < 0.1) & (t > 0.005)
    x = torch.where(
        mask.view(B, 1, 1, 1), x, x.detach()
    )

    # heading 벡터 정규화 (cos, sin 부분)
    heading = x[..., 2:4]
    heading = heading / heading.norm(dim=-1, keepdim=True).detach()
    x = torch.cat([x[..., :2], heading], dim=-1)  # [B, P+1, T+1, 4]

    # 예측 궤적 (첫 timestep 제외)에서 위치 정보만 추출
    traj = x[:, :, :, :2]  # [B, P+1, T+1, 2]

    # successive distance 계산
    diffs = traj[:, :, 1:, :] - traj[:, :, :-1, :]  # [B, P+1, T, 2]
    d = diffs.norm(dim=-1)                           # [B, P+1, T]

    # 이동 한계 거리
    # TODO: 도로 속도 제한에 따라, 바꾸기
    r_max = (100/3.6) * 0.1 #inputs['r_max']                         # scalar

    # 위반 정도 (sparsity)
    violation = F.relu((d - r_max) / r_max)         # [B, P+1, T]
    indicator = (d > r_max).float()                 # [B, P+1, T]

    # 에너지 계산: 위반 평균
    energy = violation.sum(dim=(0,1,2)) / (
        indicator.sum(dim=(0,1,2)) + _EPS
    )                                               # [1]

    # guidance 신호 (음의 지수 형태)
    reward = -torch.exp(energy)                   # [1]

    #############
# 1) reward에 대한 x의 gradient 추출
    grad_all = torch.autograd.grad(
        reward.sum(),  # 스칼라이므로 .sum()
        x,
        retain_graph=True,
        allow_unused=True
    )[0]                                     # [B, P+1, T+1, 4]

    # 2) 위치 성분(x,y)만 골라내기
    x_aux = grad_all[:, 1:, :, :2] # (1, 10, 81, 2)

    # 3) 각 agent별 heading(cos,sin)으로 회전행렬 만들기
    cos_n = x[:, 1:, :, 2]   # [B, Pn, T+1]
    sin_n = x[:, 1:, :, 3]   # [B, Pn, T+1]
    rot = torch.stack([cos_n,  sin_n,
                       -sin_n, cos_n], dim=-1)  # [B, Pn, T+1, 4]
    B, Pp1, Tp1, _ = rot.shape
    R = rot.view(B, Pp1, Tp1, 2, 2)         # [B, P+1, T+1, 2,2]

    # 4) world frame으로 회전
    world_grad = torch.einsum("bptij,bptj->bpti", R, x_aux)  # [B, P+1, T+1, 2]

    # 5) lateral 성분만 Gaussian smoothing
    lat = world_grad[..., 1].contiguous().view(-1, 1, Tp1)   # [B*(P+1),1,T+1]
    lat = F.pad(lat, (10,10), mode='replicate')
    kernel = torch.exp(-torch.linspace(-2,2,21,device=x.device)**2/4).view(1,1,21)
    lat = F.conv1d(lat, kernel)                             # [B*(P+1),1,T+1]
    lat = lat.view(B, Pp1, Tp1)                             # [B, P+1, T+1]

    # 6) longitudinal 성분은 0으로, lateral만 재조합
    grad_smooth = torch.stack([
        torch.zeros_like(lat),
        lat
    ], dim=-1)                                              # [B, P+1, T+1, 2]

    # 7) world→ego inverse 회전
    ego_grad = torch.einsum("bptji,bptj->bpti", R, grad_smooth)  # [B, P+1, T+1, 2]

    # 8) 최종 guidance: 궤적 좌표와 내적 → 각 agent별 점수, 평균 또는 원하는 방식으로 집계
    #    여기서는 P+1개 agent을 모두 평균
    guidance_per_agent = torch.sum(ego_grad.detach() * x[:, 1:, :, :2], dim=(2,3))  # [B, P+1]
    guidance = guidance_per_agent.mean(dim=1)                   # [B,]

    return 3.0 * guidance  # 스케일 맞춰서 반환