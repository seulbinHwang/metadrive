import torch
from typing import Tuple
"""
TODO
- 각 차량 (11대 중 하나) 는, `vehicle.navigation.current_ref_lanes`와 `vehicle.navigation.next_ref_lanes` 을 가지고 있습니다. 
    (List[lane] )
  - 그 2개 차선들을 넘는지 안넘는지로 guidance function을 계산해야함.)
"""


def lane_offroad_distance(
        lanes: torch.Tensor,  # shape (N, 20, 12)
        points: torch.Tensor  # shape (M, 2)
) -> torch.Tensor:
    """
    각 점(points)이 N개의 차선(lanes) 경계(좌/우)로부터 벗어난 거리(미터) 계산.
    점이 어느 차선 경계 내부에 있으면 0을 반환합니다.

    Args:
        lanes: Tensor[N, 20, 12], 차선 데이터.
               cols 0-1 = 중심선 (x,y)
               cols 4-5 = 왼쪽 경계까지 오프셋 (dx,dy)
               cols 6-7 = 오른쪽 경계까지 오프셋 (dx,dy)
        points: Tensor[M, 2], (x,y) 좌표.

    Returns:
        Tensor[M], 각 점의 off-road 거리.
    """
    device = lanes.device
    N, K, _ = lanes.shape  # N: 차선 개수, K: 최대 포인트 개수 (20)
    M = points.shape[0]  # M: 점 개수

    # 1) 차선 경계점 계산
    centers = lanes[..., 0:2]  # (N, K, 2)
    left_pts = centers + lanes[..., 4:6]  # (N, K, 2)
    right_pts = centers + lanes[..., 6:8]  # (N, K, 2)

    # 2) 모든 경계 세그먼트 A->B 준비 (좌/우 경계 모두)
    A = torch.cat([left_pts[:, :-1, :], right_pts[:, :-1, :]],
                  dim=0)  # (2N, K-1, 2)
    B = torch.cat([left_pts[:, 1:, :], right_pts[:, 1:, :]],
                  dim=0)  # (2N, K-1, 2)
    S = A.numel() // 2  # 총 세그먼트 수 = 2N*(K-1)
    A_flat = A.reshape(S, 2)  # (S, 2)
    B_flat = B.reshape(S, 2)  # (S, 2)

    # 3) 점-세그먼트 거리 계산 (벡터 연산)
    AB = B_flat - A_flat  # (S, 2)
    AP = points.unsqueeze(1) - A_flat.unsqueeze(0)  # (M, S, 2)
    AB_len2 = (AB * AB).sum(dim=1, keepdim=True).clamp_min(1e-8)  # (S,1)
    t = (AP * AB.unsqueeze(0)).sum(dim=2, keepdim=True) / AB_len2  # (M, S, 1)
    t = t.clamp(0.0, 1.0)  # (M, S, 1)
    proj = A_flat.unsqueeze(0) + t * AB.unsqueeze(0)  # (M, S, 2)
    dist = (points.unsqueeze(1) - proj).norm(dim=2)  # (M, S)
    min_dist, _ = dist.min(dim=1)  # (M,)

    # 4) 차선 내부(inside) 검출 (quad union)
    LL = left_pts[:, :-1, :]  # (N, K-1, 2)
    LN = left_pts[:, 1:, :]
    RN = right_pts[:, 1:, :]
    RL = right_pts[:, :-1, :]
    quad = torch.stack([LL, LN, RN, RL], dim=2)  # (N, K-1, 4, 2)
    edges = torch.roll(quad, -1, dims=2) - quad  # (N, K-1, 4, 2)

    # points 차원 확장 후 cross-product 계산
    rel = points.view(M, 1, 1, 1, 2) - quad.view(1, N, K - 1, 4,
                                                 2)  # (M, N, K-1, 4, 2)
    e = edges.view(1, N, K - 1, 4, 2)
    cross = e[..., 0] * rel[..., 1] - e[..., 1] * rel[..., 0]  # (M, N, K-1, 4)
    inside_quad = (cross >= 0).all(dim=4)  # (M, N, K-1)
    inside = inside_quad.any(dim=(1, 2))  # (M,)

    # 5) 내부 점은 거리 0
    offroad_dist = torch.where(inside, torch.zeros_like(min_dist), min_dist)
    return offroad_dist
