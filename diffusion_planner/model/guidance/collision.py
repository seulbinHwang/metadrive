import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters

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
    B, P, T, _ = x.shape  # (B, 11, 81, 4)

    x: torch.Tensor = x.reshape(B, P, -1, 4)
    mask_diffusion_time = (t < 0.1 and t > 0.005)
    x = torch.where(mask_diffusion_time, x, x.detach())
    # heading: (B, 11, 81, 2)
    heading = x[:, :, :, 2:].detach() / torch.norm(
        x[:, :, :, 2:].detach(), dim=-1, keepdim=True)
    x = torch.cat([x[:, :, :, :2], heading], dim=-1)  # (B, 11, 81, 4)

    ego_pred = x[:, :1, 1:, :]  # [B, 1, 80, 4]
    cos_h, sin_h = ego_pred[..., 2:3], ego_pred[..., 3:4]
    ego_pred = torch.cat([
        ego_pred[..., 0:1] + cos_h * COG_TO_REAR,
        ego_pred[..., 1:2] + sin_h * COG_TO_REAR, ego_pred[..., 2:]
    ],
                         dim=-1)  # [B, 1, 80, 4]

    neighbors_pred = x[:, 1:, 1:, :]  # [B, 10, 80, 4]

    B, Pn, T, _ = neighbors_pred.shape  # Pn = 10 , T = 80

    predictions = torch.cat([ego_pred, neighbors_pred.detach()],
                            dim=1)  # [B, 11, 80, 4]

    lw = torch.cat([
        torch.tensor(ego_size, device=predictions.device)[None, None, :].repeat(
            B, 1, 1), inputs["neighbor_agents_past"][:, :Pn, -1, [7, 6]]
    ],
                   dim=1)  # [1, 11, 2]
    bbox = torch.cat(
        [predictions,
         lw.unsqueeze(2).expand(-1, -1, T, -1) + INFLATION],
        dim=-1)  # [1, 11, 80, 6]
    bbox = center_rect_to_points(bbox.reshape(-1, 6)).reshape(
        B, Pn + 1, T, 4, 2)  # (1, 11, 80, 4, 2)
    neighbor_current_mask = inputs["neighbor_current_mask"]  # [B, 10]
    ego_bbox = bbox[:, :1, :, :, :].expand(-1, Pn, -1, -1,
                                           -1)[~neighbor_current_mask].reshape(
                                               -1, 4, 2)  # [800, 4, 2]
    neighbor_bbox = bbox[:, 1:, :, :, :][~neighbor_current_mask].reshape(
        -1, 4, 2)  # [800, 4, 2]

    distances = batch_signed_distance_rect(ego_bbox, neighbor_bbox)  # [800]
    clip_distances = torch.maximum(1 - distances / CLIP_DISTANCE,
                                   torch.tensor(
                                       0.0, device=distances.device))  # [800]

    energy = (torch.sum(clip_distances[clip_distances > 1]) / (torch.sum(
        (clip_distances[clip_distances > 1].detach() > 0).float()) + 1e-5) +
         torch.sum(clip_distances[clip_distances <= 1]) / (torch.sum(
             (clip_distances[clip_distances <= 1].detach() > 0).float()) + 1e-5)
        )  # []
    reward = -energy.exp()  # []

    x_aux = torch.autograd.grad(reward.sum(),
                                x,
                                retain_graph=True,
                                allow_unused=True)[0]  # [B, 11, 81, 4]
    x_aux = x_aux[:, 0, :, :2]  # [1, 81, 2]


    T += 1
    x_mat = torch.einsum(
        "btd,nd->btn", x[:, 0, :, 2:],
        torch.tensor([[1., 0], [0, 1], [0, -1], [1, 0]],
                     device=x.device)).reshape(B, T, 2, 2) # [1, 81, 2, 2]

    x_aux = torch.einsum("btij,btj->bti", x_mat, x_aux) # [1, 81, 2]

    # ----------------------------------------------------------
    # x_aux: [B, T, 2]  ←  world-frame grad (longitudinal, lateral)
    # 이 블록은 ① 전‧후(long) 성분 제거 ② 좌‧우(lat) 성분 부드럽게(Gauss) 필터링
    # ----------------------------------------------------------

    # ─── ① Longitudinal(전‧후) 축: 아예 제거 ──────────────────────
    # decay_vec: ([1.0000, 0.9876, 0.9753,  ..., 0.3772, 0.3725, 0.3679])
    decay_vec: torch.Tensor = (
        -torch.linspace(0, 1, T, device=x.device)).exp()  # shape (81,)
    """
    # decay_vec: 
    ([1.0000, 0.9876, 0.9753,  ..., 0.3772, 0.3725, 0.3679]),
    ...
    ([1.0000, 0.9876, 0.9753,  ..., 0.3772, 0.3725, 0.3679]),

tri_mask: 
tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [1.0000, 0.9876, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [1.0000, 0.9876, 0.9753,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [1.0000, 0.9876, 0.9753,  ..., 0.3772, 0.0000, 0.0000],
        [1.0000, 0.9876, 0.9753,  ..., 0.3772, 0.3725, 0.0000],
        [1.0000, 0.9876, 0.9753,  ..., 0.3772, 0.3725, 0.3679]],

    """
    decay_vec = decay_vec.unsqueeze(0).repeat(T, 1) # shape (81, 81)
    tri_mask: torch.Tensor = torch.tril(decay_vec)  # shape (81, 81)
    long_grad: torch.Tensor = torch.einsum(  # shape (B, T)
        'bt,ti->bi',  # (1,81) · (81,81) → (1, 81) # gradient를 전부 더하는거야. (내 자신 포함 뒷쪽 에 있는 애들)
        x_aux[..., 0],  # longitudinal grad # (1, 81)
        tri_mask,
    )
    long_grad = torch.zeros_like(long_grad)  # → 완전히 0 # (1, 81)

    # ─── ② Lateral(좌‧우) 축: 1-D Gaussian smoothing ──────────────
    kernel_size: int = 21
    half_k: int = kernel_size // 2 # 10
    """a
    ([
    -1, -0.81, -0.64, -0.49, -0.36, -0.25, -0.16, -0.09, -0.04, -0.01, 
    -0.00,
    -0.01, -0.04, -0.09, -0.16, -0.25, -0.36, -0.49, -0.64, -0.81, -1.00])

gauss_kernel: tensor([[[
            0.3679, 0.4449, 0.5273, 0.6126, 0.6977, 0.7788, 0.8521, 0.9139, 0.9608, 0.9900, 
            1.0000, 
          0.9900, 0.9608, 0.9139, 0.8521, 0.7788, 0.6977, 0.6126, 0.5273, 0.4449, 0.3679
          ]]], device='cuda:0')

    """
    a = -torch.linspace(-2, 2, kernel_size, device=x.device)**2 / 4 # (21,)
    gauss_kernel = a.exp().view(1, 1, -1)  # shape (1,1,21)
    # gauss_kernel = gauss_kernel / gauss_kernel.sum()  # 합이 1이 되도록 정규화

    y_x_aux = x_aux[:, None, :, 1] # (1, 1, 81)
    lat_grad_padded: torch.Tensor = F.pad(  # shape (B,1,T+2*half_k) # (1, 1, 101)
        y_x_aux,
        (half_k, half_k),  # replicate pad
        mode='replicate',
    )
    # gaussian weighted sum.
    lat_grad = F.conv1d(lat_grad_padded, gauss_kernel)  # shape (1, 1, 81)
    lat_grad = lat_grad[:, 0] # (1, 81)
    # ─── ③ 두 축을 다시 결합 → [B, T, 2] ─────────────────────────
    x_aux_filtered: torch.Tensor = torch.stack([long_grad, lat_grad],
                                               dim=2)  # (B,T,2)

    ############

    x_aux = torch.einsum("btji,btj->bti", x_mat, x_aux_filtered)  # [B, T, 2]

    reward = torch.sum(x_aux.detach() * x[:, 0, :, :2], dim=(1, 2))

    return 3.0 * reward
