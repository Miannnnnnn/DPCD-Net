import torch
import random
import torch.nn as nn
import numpy as np

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def get_derivative_kernels(dx, dy):
    
    # 一阶导数核
    kernel_x = torch.tensor([[[[[-1, 0, 1]]]]], dtype=torch.float32).expand(1, 1, 1, 1, 3) / (2 * dx)
    kernel_y = torch.tensor([[[[[-1], [0], [1]]]]], dtype=torch.float32).expand(1, 1, 1, 3, 1) / (2 * dy)

    # 二阶导数核
    kernel_xx = torch.tensor([[[[[1, -2, 1]]]]], dtype=torch.float32).expand(1, 1, 1, 1, 3) / (dx ** 2)
    kernel_yy = torch.tensor([[[[[1], [-2], [1]]]]], dtype=torch.float32).expand(1, 1, 1, 3, 1) / (dy ** 2)

    return kernel_x, kernel_y, kernel_xx, kernel_yy


def compute_burgers_residual(u, v, dx=27000, dy=27000, nu=250000.0):
    """
    稳态Burgers方程残差计算（不包含时间导数）

    方程形式：
    u∂u/∂x + v∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)
    u∂v/∂x + v∂v/∂y = ν(∂²v/∂x² + ∂²v/∂y²)

    Args:
        u, v: 风速分量 [batch, channel, time, height, width]
        dx, dy: 网格间距（米）
        nu: 涡动粘度系数（m²/s）
    """

    
    kernel_x, kernel_y, kernel_xx, kernel_yy = get_derivative_kernels(dx, dy)

    device = u.device
    kernel_x = kernel_x.to(device)
    kernel_y = kernel_y.to(device)
    kernel_xx = kernel_xx.to(device)
    kernel_yy = kernel_yy.to(device)

    # 计算一阶空间导数
    u_x = F.conv3d(u, kernel_x, padding=(0, 0, 1))
    u_y = F.conv3d(u, kernel_y, padding=(0, 1, 0))
    v_x = F.conv3d(v, kernel_x, padding=(0, 0, 1))
    v_y = F.conv3d(v, kernel_y, padding=(0, 1, 0))

    # 计算二阶空间导数
    u_xx = F.conv3d(u, kernel_xx, padding=(0, 0, 1))
    u_yy = F.conv3d(u, kernel_yy, padding=(0, 1, 0))
    v_xx = F.conv3d(v, kernel_xx, padding=(0, 0, 1))
    v_yy = F.conv3d(v, kernel_yy, padding=(0, 1, 0))

    # 稳态Burgers方程残差
    residual_u = u * u_x + v * u_y - nu * (u_xx + u_yy)
    residual_v = u * v_x + v * v_y - nu * (v_xx + v_yy)

    return torch.mean(residual_u.pow(2) + residual_v.pow(2))


def burgers_loss(img_real, img_out, dx=27000, dy=27000, nu=250000.0):
    """
    Args:
        img_real: 真实风速场 [batch, 2, time, height, width]
        img_out: 预测风速场 [batch, 2, time, height, width]
        dx, dy: 网格间距（米），ERA5 0.25度 ≈ 27km
        nu: 涡动粘度系数（m²/s）
    """
    
    u_out = img_out[:, 0:1, :, :, :]  # [batch, 1, time, height, width]
    v_out = img_out[:, 1:2, :, :, :]

    physics_loss = compute_burgers_residual(u_out, v_out, dx, dy, nu)

    return physics_loss


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def toNE(pred_traj,pred_Me):
    # 0  经度  1纬度
    pred_traj[:, :,0] = pred_traj[:, :,0] / 10 * 500 + 1300
    pred_traj[:,:,1] = pred_traj[:,:,1] / 6 * 300 + 300
    # 0 气压 1 风速
    pred_Me[:, :, 0] = pred_Me[:, :, 0]* 50 + 960
    pred_Me[:, :, 1] = pred_Me[:, :, 1] * 25 + 40
    return pred_traj,pred_Me

def calculate_haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(lambda x: x * np.pi / 180.0, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine 公式
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
   
    c = 2 * torch.atan2(torch.sqrt(torch.clamp(a, 0, 1)), torch.sqrt(torch.clamp(1 - a, 0, 1)))
    
    return 6371.0 * c

def trajectory_displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):

    lon_p, lat_p = pred_traj.permute(1, 0, 2)[:, :, 0] / 10.0, pred_traj.permute(1, 0, 2)[:, :, 1] / 10.0
    lon_g, lat_g = pred_traj_gt.permute(1, 0, 2)[:, :, 0] / 10.0, pred_traj_gt.permute(1, 0, 2)[:, :, 1] / 10.0

    loss = calculate_haversine(lon_p, lat_p, lon_g, lat_g)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def trajectory_diff(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    
    lat_p, lon_p = pred_traj.permute(1, 0, 2)[:, :, 0] / 10.0, pred_traj.permute(1, 0, 2)[:, :, 1] / 10.0
    lat_g, lon_g = pred_traj_gt.permute(1, 0, 2)[:, :, 0] / 10.0, pred_traj_gt.permute(1, 0, 2)[:, :, 1] / 10.0

    loss = calculate_haversine(lon_p, lat_p, lon_g, lat_g)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def value_diff(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj.permute(1, 0, 2)-pred_traj_gt.permute(1, 0, 2)

    # loss = loss**2
    # loss = torch.sqrt(loss[:,:,0]+loss[:,:,1])
    # if consider_ped is not None:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    # else:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def value_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = torch.abs((pred_traj.permute(1, 0, 2)-pred_traj_gt.permute(1, 0, 2)))

    # loss = loss**2
    # loss = torch.sqrt(loss[:,:,0]+loss[:,:,1])
    # if consider_ped is not None:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    # else:
    #     loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
