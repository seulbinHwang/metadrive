import torch
import torch.nn.functional as F

# 물리적 최대 이동 거리 (delta_t 동안)
# inputs에서 r_max를 받아 사용합니다.
_EPS = 1e-6


def feasible_guidance_fn(x: torch.Tensor, t: torch.Tensor, cond, inputs: dict,
                         *args, **kwargs) -> torch.Tensor:
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
    B, P, T, _ = x.shape  # (B, 11, 81, 4)

    x: torch.Tensor = x.reshape(B, P, -1, 4)
    mask_diffusion_time = (t < 0.1 and t > 0.005)
    x = torch.where(mask_diffusion_time, x, x.detach())



    # heading: (B, 11, 81, 2)
    heading = x[:, :, :, 2:].detach() / torch.norm(
        x[:, :, :, 2:].detach(), dim=-1, keepdim=True)
    x = torch.cat([x[:, :, :, :2], heading], dim=-1)  # (B, 11, 81, 4)

    neighbor_traj = x[:, 1:, :, :2]  # neighbor_traj:(B, 10, 81, 2) # 궤적 좌표 (x,y)
    # remove neighbor_current_mask 값이 True 에 해당하는 agent는 -> traj에서 제거해야 함
    diffs_norm = []

    neighbor_current_mask = inputs[
        "neighbor_current_mask"]  # (B, 10) # 차량이 없는 경우 True
    if ~neighbor_current_mask.sum() == 0:                         # 이웃이 없다면 skip
        # guidance 불필요 — 그래프 유지용 dummy 값 반환
        return (x[..., 0] * 0).sum()
    for b in range(B):
        keep_mask = ~neighbor_current_mask[b]  # (10,)  존재하는 에이전트만 True
        traj_present = neighbor_traj[b][keep_mask]  # (N_b, 81, 2)
        diff = traj_present[:, 1:2, :] - traj_present[:, 0:1, :]  # (N_b, 80, 2)
        d = diff.norm(dim=-1)  # (N_b, 80)
        diffs_norm.append(d)
    # diffs_norm: (N_b_0 + N_b_1 + ... + N_b_B, 80)
    diffs_norm = torch.cat(diffs_norm, dim=0)  # (N, 80)
    # diffs_norm: (N, 80) -> (N * 80) 1차원으로 변경
    diffs_norm = diffs_norm.view(-1)  # (N * 80)

    # 이동 한계 거리
    r_max = (30. / 3.6) * 0.1  #inputs['r_max']                         # scalar

    # 위반 정도 (sparsity)
    violation = F.relu((diffs_norm - r_max) / r_max)  # (N * 80)
    violation = torch.minimum(violation, torch.tensor(10.0,
                         device=violation.device))
    indicator = (diffs_norm > r_max).float()  # (N * 80)


    # 에너지 계산: 위반 평균
    energy = violation.sum() / (indicator.sum() + _EPS)  # []

    # guidance 신호 (음의 지수 형태)
    reward = -energy.exp()  # []

    #############
    # 1) reward에 대한 x의 gradient 추출
    x_aux = torch.autograd.grad(
        reward.sum(),  # 스칼라이므로 .sum()
        x,
        retain_graph=True,
        allow_unused=True)[0] # [B, 11, 81, 4]

    # 2) 위치 성분(x,y)만 골라내기
    x_aux = x_aux[:, 1:, 1:2, :2]  # (B, 10, 80, 2)
    reward = torch.sum(x_aux.detach() * x[:, 1:, 1:2, :2], dim=(1, 2, 3))  # [B]
    return 3.0 * reward  # 스케일 맞춰서 반환

    # # 3) 각 agent별 heading(cos,sin)으로 회전행렬 만들기
    # cos_n = x[:, 1:, :, 2]  # [B, Pn, T+1]
    # sin_n = x[:, 1:, :, 3]  # [B, Pn, T+1]
    # rot = torch.stack([cos_n, sin_n, -sin_n, cos_n], dim=-1)  # [B, Pn, T+1, 4]
    # B, Pp1, Tp1, _ = rot.shape
    # R = rot.view(B, Pp1, Tp1, 2, 2)  # [B, P+1, T+1, 2,2]
    #
    # # 4) world frame으로 회전
    # world_grad = torch.einsum("bptij,bptj->bpti", R, x_aux)  # [B, P+1, T+1, 2]
    #
    # # 5) lateral 성분만 Gaussian smoothing
    # lat = world_grad[..., 1].contiguous().view(-1, 1, Tp1)  # [B*(P+1),1,T+1]
    # lat = F.pad(lat, (10, 10), mode='replicate')
    # kernel = torch.exp(-torch.linspace(-2, 2, 21, device=x.device)**2 / 4).view(
    #     1, 1, 21)
    # lat = F.conv1d(lat, kernel)  # [B*(P+1),1,T+1]
    # lat = lat.view(B, Pp1, Tp1)  # [B, P+1, T+1]
    #
    # # 6) longitudinal 성분은 0으로, lateral만 재조합
    # grad_smooth = torch.stack([torch.zeros_like(lat), lat],
    #                           dim=-1)  # [B, P+1, T+1, 2]
    #
    # # 7) world→ego inverse 회전
    # ego_grad = torch.einsum("bptji,bptj->bpti", R,
    #                         grad_smooth)  # [B, P+1, T+1, 2]
    #
    # # 8) 최종 guidance: 궤적 좌표와 내적 → 각 agent별 점수, 평균 또는 원하는 방식으로 집계
    # #    여기서는 P+1개 agent을 모두 평균
    # guidance_per_agent = torch.sum(ego_grad.detach() * x[:, 1:, :, :2],
    #                                dim=(2, 3))  # [B, P+1]
    # guidance = guidance_per_agent.mean(dim=1)  # [B,]
