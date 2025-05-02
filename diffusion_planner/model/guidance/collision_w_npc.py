import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, \
    get_pacifica_parameters

ego_size = [get_pacifica_parameters().length, get_pacifica_parameters().width]

COG_TO_REAR = 1.67
CLIP_DISTANCE = 1.0
INFLATION = 1.0


def batch_signed_distance_rect(rect1, rect2):
    '''
    rect1: [B, 4, 2]
    rect2: [B, 4, 2]

    return [B] (signed distance between two rectangles)
    '''
    B, _, _ = rect1.shape
    norm_vec = torch.stack([
        rect1[:, 0] - rect1[:, 1], rect1[:, 1] - rect1[:, 2],
        rect2[:, 0] - rect2[:, 1], rect2[:, 1] - rect2[:, 2]
    ],
                           dim=1)  # [B, 4, 2]
    norm_vec = norm_vec / torch.norm(norm_vec, dim=2, keepdim=True)

    proj1 = torch.einsum('bij,bkj->bik', norm_vec,
                         rect1)  # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]
    proj1_min, proj1_max = proj1.min(dim=2)[0], proj1.max(
        dim=2)[0]  # [B, 4] [B, 4]

    proj2 = torch.einsum('bij,bkj->bik', norm_vec,
                         rect2)  # [B, 4, 2] * [B, 4, 2] -> [B, 4, 4]
    proj2_min, proj2_max = proj2.min(dim=2)[0], proj2.max(
        dim=2)[0]  # [B, 4] [B, 4]

    overlap = torch.cat([proj1_min - proj2_max, proj2_min - proj1_max],
                        dim=1)  # [B, 8]

    positive_distance = torch.where(overlap < 0, 1e5, overlap)

    is_overlap = (overlap < 0).all(dim=1)
    distance = torch.where(is_overlap,
                           overlap.max(dim=1).values,
                           positive_distance.min(dim=1).values)

    return distance


def center_rect_to_points(rect):
    '''
    rect: [B, 6] (x, y, cos_h, sin_h, l, w)

    return [B, 4, 2] (4 points of the rectangle)
    '''

    B, _ = rect.shape
    xy, cos_h, sin_h, lw = rect[:, :2], rect[:, 2], rect[:, 3], rect[:, 4:]

    rot = torch.stack([cos_h, -sin_h, sin_h, cos_h],
                      dim=1).reshape(-1, 2, 2)  # [B, 2, 2]
    lw = torch.einsum(
        'bj,ij->bij', lw,
        torch.tensor([[1., 1], [-1, 1], [-1, -1], [1, -1]], device=lw.device) /
        2)  # [B, 2] * [4, 2] -> [B, 4, 2]
    lw = torch.einsum('bij,bkj->bik', lw,
                      rot)  # [B, 4, 2] * [B, 2, 2] -> [B, 4, 2]

    rect = xy[:, None, :] + lw  # [B, 4, 2]

    return rect


def collision_guidance_fn(x, t, cond, inputs, *args, **kwargs) -> torch.Tensor:
    """
    x: [B * Pn+1, T + 1, 4]
    t: [B, 1],
    inputs: Dict[str, torch.Tensor]
    """
    B, P, T, _ = x.shape
    neighbor_current_mask = inputs["neighbor_current_mask"]  # (B, 10)
    all_neighbor_current_mask = inputs["all_neighbor_current_mask"]  # (B, 32)
    predicted_agents_num = x.shape[1] - 1  # Pn
    assert (predicted_agents_num == 10)

    # ==== 추가된 부분: nearest 10대/나머지 분리 ====
    # inputs["neighbor_agents_past"]: [B, N=32, 21, 11]
    # 현재 시점 feature (x,y,cos,sin,vx,vy,width,length,…)
    neighbor_agents_current = inputs[
        "neighbor_agents_past"][:, :, -1, :]  # [B, N, 11]
    leftover_agents_current = neighbor_agents_current[:,
                                                      predicted_agents_num:, :]  # [B, N-10, 11]
    far_count = leftover_agents_current.size(1)  # 22
    half_T = (T - 1) // 2  # 80 -> 40
    assert half_T == 40
    # ==== 추가된 부분: 나머지 차량 constant-velocity future 생성 ====
    pos = leftover_agents_current[..., 0:2]  # [B,22,2]
    vel = leftover_agents_current[..., 4:6]  # [B,22,2]
    heading = leftover_agents_current[..., 2:4]  # [B,22,2]
    # ==== 수정된 부분: 시간 간격 Δt 적용 ====
    delta_t = 0.1  # 인접 점 간 시간 간격 (s)
    a = torch.arange(1, T, device=x.device)
    times = a.view(1, 1, T - 1, 1) * delta_t  # [1,1,80,1]
    pos_fut = pos.unsqueeze(2) + vel.unsqueeze(2) * times  # [1,22,80,2]
    cos_fut = heading[..., 0].unsqueeze(2).expand(-1, -1, T - 1)  # [B,22,80]
    sin_fut = heading[..., 1].unsqueeze(2).expand(-1, -1, T - 1)  # [B,22,80]
    neighbor_leftover_agents_future = torch.cat(
        [pos_fut, cos_fut.unsqueeze(-1),
         sin_fut.unsqueeze(-1)], dim=-1)  # [B,22,80,4]
    # ============================
    x: torch.Tensor = x.reshape(B, P, -1, 4)
    mask_diffusion_time = (t < 0.1 and t > 0.005)
    x = torch.where(mask_diffusion_time, x, x.detach())

    x = torch.cat([
        x[:, :, :, :2], x[:, :, :, 2:].detach() /
        torch.norm(x[:, :, :, 2:].detach(), dim=-1, keepdim=True)
    ],
                  dim=-1)  # [B, P + 1, T, 4]

    ego_pred = x[:, :1, 1:, :]  # [B, 1, T, 4]
    cos_h, sin_h = ego_pred[..., 2:3], ego_pred[..., 3:4]
    ego_pred = torch.cat([
        ego_pred[..., 0:1] + cos_h * COG_TO_REAR,
        ego_pred[..., 1:2] + sin_h * COG_TO_REAR, ego_pred[..., 2:]
    ],
                         dim=-1)

    neighbors_pred = x[:, 1:, 1:, :]  # [B, P, T, 4]

    B, Pn, T, _ = neighbors_pred.shape

    # predictions = torch.cat([ego_pred, neighbors_pred.detach()],
    #                         dim=1)  # [B, P + 1, T, 4]
    # ==== 기존 predictions 확장 ====
    #  - 원본: torch.cat([ego_pred, neighbors_pred.detach()], dim=1)
    predictions = torch.cat(
        [
            ego_pred,
            neighbors_pred.detach(),
            neighbor_leftover_agents_future  # [B, 1+Pn+22, T, 4]
        ],
        dim=1)  # (1, 33, 80, 4)
    # ==== 기존 lw 확장 ====
    #  - 원본: lw = torch.cat([ego_size, inputs["neighbor_agents_past"][:,:Pn,-1,[7,6]]], dim=1)
    lw = torch.cat([
        torch.tensor(ego_size, device=predictions.device)[None, None, :].repeat(
            B, 1, 1), inputs["neighbor_agents_past"][:, :Pn, -1, [7, 6]]
    ],
                   dim=1)  # [1, 11, 2)
    leftover_lw = inputs["neighbor_agents_past"][:, predicted_agents_num:, -1,
                                                 [7, 6]]  # [1,22,2]
    lw = torch.cat(
        [
            lw,  # 이전까지 [B, 1+Pn, 2]
            leftover_lw  # 추가 -> [B, 1+Pn+22, 2]
        ],
        dim=1)  # (1, 33, 2)
    bbox = torch.cat(
        [predictions,
         lw.unsqueeze(2).expand(-1, -1, T, -1) + INFLATION],
        dim=-1)  # (1, 33, 80, 6)
    bbox = center_rect_to_points(bbox.reshape(-1, 6)).reshape(
        B, 1 + Pn + far_count, T, 4, 2)  # (1, 33, 80, 4, 2)
    # ==== 추가된 부분: near/far bounding‐box flattening ====
    # 원본은 전체 Pn에 대해 전부 T timestep을 비교했지만,
    # → near 10대는 T 전체, far 22대는 half_T(40)만 비교
    far_nbr = bbox[:, 1 +
                   predicted_agents_num:, :half_T, :, :]  # [1, 22, 40, 4, 2]
    distances = []
    # TODO: remove for loop?
    for idx in range(1, 1 + predicted_agents_num):  # 0 ~ 10
        near_ego = bbox[:,
                        idx:idx + 1, :, :, :].expand(-1, predicted_agents_num,
                                                     -1, -1,
                                                     -1)  # (1, 10, 80, 4, 2)
        far_ego = bbox[:, idx:idx + 1, :half_T, :, :].expand(
            -1, far_count, half_T, -1, -1)  # [1, 22, 40, 4, 2]
        a = bbox[:, :idx, :, :, :]
        b = bbox[:, idx + 1:1 + predicted_agents_num, :, :, :]
        near_nbr = torch.cat([a, b], dim=1)  # [1, 10, 80, 4, 2]
        a = all_neighbor_current_mask[:, :idx - 1]
        b = all_neighbor_current_mask[:, idx:predicted_agents_num]
        near_mask = torch.cat([a, b], dim=1)  # [B, 10]
        true_padding = torch.ones_like(near_mask[:, :1],
                                       device=near_mask.device)  # [B, 1]
        # pad True in front.
        near_mask = torch.cat([true_padding, near_mask], dim=1)  # [B, 11]
        ego_near_flat = near_ego[~near_mask].reshape(-1, 4, 2)
        nbr_near_flat = near_nbr[~near_mask].reshape(-1, 4, 2)

        far_mask = all_neighbor_current_mask[:, predicted_agents_num:]  # [B,22]
        ego_far_flat = far_ego[~far_mask].reshape(-1, 4, 2)
        nbr_far_flat = far_nbr[~far_mask].reshape(-1, 4, 2)

        ego_bbox = torch.cat([ego_near_flat, ego_far_flat], dim=0)  # (_, 4, 2)
        neighbor_bbox = torch.cat([nbr_near_flat, nbr_far_flat],
                                  dim=0)  # (_, 4, 2)
        a_distances = batch_signed_distance_rect(ego_bbox, neighbor_bbox)  # (_)
        distances.append(a_distances)
    distances = torch.cat(distances, dim=0)  # [N]
    ######################
    clip_distances = torch.maximum(1 - distances / CLIP_DISTANCE,
                                   torch.tensor(0.0, device=distances.device))

    reward = -(torch.sum(clip_distances[clip_distances > 1]) / (torch.sum(
        (clip_distances[clip_distances > 1].detach() > 0).float()) + 1e-5) +
               torch.sum(clip_distances[clip_distances <= 1]) / (torch.sum(
                   (clip_distances[clip_distances <= 1].detach() > 0).float()) +
                                                                 1e-5)).exp()
    T += 1
    # 1) ego 대신 나머지 Pn대 차량에 대한 gradient 계산
    grad_all = torch.autograd.grad(
        reward.sum(),  # scalar reward의 합
        x,  # wrt 전체 trajectory tensor
        retain_graph=True,
        allow_unused=True)[0]  # 결과 shape: [B, P+1, T+1, 4] # (1, 11, 81, 4)
    # 2) ego(0번) 제외하고 위치 성분만 추출 ([B, Pn, T+1, 2])
    x_aux = grad_all[:, 1:, :, :2]  # (1, 10, 81, 2)
    # 3) 각 neighbor 차량별 heading(cos, sin)으로 회전 행렬 생성
    cos_n = x[:, 1:, :, 2]  # [B, Pn, T+1]
    sin_n = x[:, 1:, :, 3]  # [B, Pn, T+1]
    rot = torch.stack([cos_n, sin_n, -sin_n, cos_n], dim=-1)  # [B, Pn, T+1, 4]
    B, Pn, Tp1, _ = rot.shape
    x_mat = rot.view(B, Pn, Tp1, 2, 2)  # [B, Pn, T+1, 2, 2] # (1, 10, 81, 2, 2)
    # 4) world frame으로 회전
    # (1, 10, 81, 2, 2) @ (1, 10, 81, 2) -> (1, 10, 81, 2)
    x_aux = torch.einsum("bptij,bptj->bpti", x_mat, x_aux)  # [B, Pn, T+1, 2]
    # 5) lateral(y) 성분만 Gaussian smoothing
    x_lat = x_aux[..., 1].contiguous().view(B * Pn, 1, Tp1)  # [B*Pn, 1, T+1]
    x_lat = F.pad(x_lat, (10, 10), mode='replicate')
    kernel = torch.exp(-torch.linspace(-2, 2, 21, device=x.device)**2 / 4).view(
        1, 1, 21)
    x_lat = F.conv1d(x_lat, kernel)  # [B*Pn, 1, T+1]
    x_lat = x_lat.view(B, Pn, Tp1)  # [B, Pn, T+1]

    # 6) longitudinal(x) 성분은 0으로, lateral 성분만 재조합
    # x_aux: (1, 10, 81, 2)
    x_aux = torch.stack(
        [
            torch.zeros_like(x_lat),  # longitudinal
            x_lat  # smoothed lateral
        ],
        dim=-1)  # [B, Pn, T+1, 2]
    # 7) world→ego inverse 회전
    # x_aux: (1, 10, 81, 2)
    x_aux = torch.einsum("bptji,bptj->bpti", x_mat, x_aux)  # [B, Pn, T+1, 2]
    # 8) 각 차량별 reward 계산 후 평균
    # reward: (B, 10)
    reward_n = torch.sum(x_aux.detach() * x[:, 1:, :, :2],
                         dim=(2, 3))  # [B, Pn]
    # reward: (B)
    reward = reward_n.mean(dim=1)  # [B]
    return 3.0 * reward
