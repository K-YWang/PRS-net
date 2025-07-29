# symmetry_loss.py
# Symmetry losses: reflection/rotation consistency + orthogonality regularization.

from typing import Dict
import torch
import torch.nn as nn


def quat_conjugate(q: torch.Tensor) -> torch.Tensor: # 四元数的共轭函数
    # (...,4) -> (...,4)
    return torch.stack([ q[..., 0],
                        -q[..., 1],
                        -q[..., 2],
                        -q[..., 3]], dim=-1)

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor: # 四元数乘法
    """
    Hamilton product for quaternions.
    q1, q2: (...,4)
    """
    w1, x1, y1, z1 = q1.unbind(-1) # 沿最后一维拆分
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_rotate(points: torch.Tensor, quat: torch.Tensor) -> torch.Tensor: # 旋转的结果
    """
    points: (B,P,3)
    quat:   (B,Kq,4)  -> broadcast over P
    returns: (B,Kq,P,3)
    """
    B, P, _ = points.shape
    Kq = quat.size(1)
    q = quat.unsqueeze(2).expand(B, Kq, P, 4)              # (B,Kq,P,4)
    zeros = torch.zeros(B, 1, P, 1, device=points.device, dtype=points.dtype)
    p_aug = torch.cat([zeros.expand(-1, Kq, -1, -1), points.unsqueeze(1).expand(-1, Kq, -1, -1)], dim=-1)  # (B,Kq,P,4)
    q_conj = quat_conjugate(q)
    qp = quat_mul(q, p_aug)
    qpq = quat_mul(qp, q_conj)
    return qpq[..., 1:4]  # (B,Kq,P,3)

def plane_reflect(points: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """
    points: (B,P,3)
    planes: (B,Kp,4) [a,b,c,d]
    returns: (B,Kp,P,3)
    """
    B, P, _ = points.shape
    Kp = planes.size(1)
    n = planes[..., :3]                                  # (B,Kp,3)
    d = planes[..., 3:4]                                 # (B,Kp,1)
    n_norm = n.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)  # (B,Kp,1)
    n_rep = n.unsqueeze(2).expand(B, Kp, P, 3)
    d_rep = d.unsqueeze(2).expand(B, Kp, P, 1)
    n_norm_rep = n_norm.unsqueeze(2).expand(B, Kp, P, 1)
    numer = (points.unsqueeze(1) * n_rep).sum(dim=-1, keepdim=True) + d_rep  # (B,Kp,P,1)
    scale = 2.0 * numer / n_norm_rep
    out = points.unsqueeze(1) - scale * (n_rep / n_norm_rep)
    return out  # (B,Kp,P,3)

def closest_cell_indices(points: torch.Tensor, grid_bound: float, grid_size: int) -> torch.Tensor:
    """
    points: (...,3) in [-grid_bound, grid_bound]
    returns: (...,3) long idx in [0, grid_size-1]
    """
    grid_min = -grid_bound + grid_bound / grid_size
    inds = (points - grid_min) * grid_size / (2 * grid_bound)
    inds = inds.round().clamp_(0, grid_size - 1).long()
    return inds

def gather_cp_and_mask(trans_points: torch.Tensor, cp: torch.Tensor, voxel: torch.Tensor,
                       grid_size: int, grid_bound: float):
    """
    trans_points: (B,K, P,3)
    cp:          (B, gs^3, 3)
    voxel:       (B, 1, gs, gs, gs)  with occupancy in {0,1}
    returns: closest_points (B,K,P,3), mask (B,K,P,1)
    """
    B, K, P, _ = trans_points.shape
    idx3 = closest_cell_indices(trans_points, grid_bound, grid_size)  # (B,K,P,3)
    # linearize: (x,y,z) -> x*gs^2 + y*gs + z
    factors = trans_points.new_tensor([grid_size**2, grid_size, 1], dtype=torch.long)
    lin_idx = (idx3 * factors).sum(dim=-1)                            # (B,K,P)
    voxel_flat = voxel.view(B, -1)                                    # (B, gs^3)
    occ = torch.gather(voxel_flat.unsqueeze(1).expand(B, K, -1), 2, lin_idx)  # (B,K,P)
    mask = (1.0 - occ).unsqueeze(-1)                                  # (B,K,P,1)

    # gather closest points
    lin_idx3 = lin_idx.unsqueeze(-1).expand(B, K, P, 3)               # (B,K,P,3)
    cp_exp = cp.unsqueeze(1).expand(B, K, -1, 3)                      # (B,K,gs^3,3)
    closest = torch.gather(cp_exp, 2, lin_idx3)                       # (B,K,P,3)
    return closest, mask

class SymmetryLoss(nn.Module):
    """
    1) Reflection consistency: reflect points by predicted planes -> distance to closestPoints (cp), masked by empty voxels.
    2) Rotation consistency:   rotate points by predicted quats  -> distance to cp, masked
    3) Regularization (separate module below)
    """
    def __init__(self, grid_size: int = 32, grid_bound: float = 0.5,
                 ref_weight: float = 1.0, rot_weight: float = 1.0):
        super().__init__()
        self.gs = int(grid_size)
        self.gb = float(grid_bound)
        self.ref_w = float(ref_weight)
        self.rot_w = float(rot_weight)

    def forward(self, points: torch.Tensor, cp: torch.Tensor, voxel: torch.Tensor,
                planes: torch.Tensor, quats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        points: (B,P,3)
        cp:     (B, gs^3, 3)
        voxel:  (B,1,gs,gs,gs)
        planes: (B,Kp,4)
        quats:  (B,Kq,4)
        """
        # Reflect & rotate
        ref_pts = plane_reflect(points, planes)     # (B,Kp,P,3)
        rot_pts = quat_rotate(points, quats)        # (B,Kq,P,3)

        # Distances
        ref_cp, ref_mask = gather_cp_and_mask(ref_pts, cp, voxel, self.gs, self.gb)
        rot_cp, rot_mask = gather_cp_and_mask(rot_pts, cp, voxel, self.gs, self.gb)

        ref_dist = ((ref_pts - ref_cp) * ref_mask).pow(2).sum(dim=-1)   # (B,Kp,P)
        rot_dist = ((rot_pts - rot_cp) * rot_mask).pow(2).sum(dim=-1)   # (B,Kq,P)

        # Mean over points, sym ops, batch
        loss_ref = ref_dist.mean() * self.ref_w
        loss_rot = rot_dist.mean() * self.rot_w

        return {"loss_ref": loss_ref, "loss_rot": loss_rot}


class RegularizationLoss(nn.Module):
    """
    Encourages orthogonality across predicted plane normals and quaternion vector parts.
    """
    def __init__(self, weight_plane: float = 1.0, weight_rot: float = 1.0):
        super().__init__()
        self.wp = float(weight_plane)
        self.wr = float(weight_rot)

    def forward(self, planes: torch.Tensor, quats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        planes: (B,Kp,4)
        quats:  (B,Kq,4)
        """
        B = planes.size(0) if planes.numel() else quats.size(0)
        device = planes.device if planes.numel() else quats.device
        I = torch.eye(3, device=device)

        # planes
        reg_plane = torch.zeros((), device=device)
        if planes.numel():
            n = planes[..., :3]                                         # (B,Kp,3)
            n = n / (n.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            X = n.transpose(1, 2)                                       # (B,3,Kp)
            G = torch.matmul(X, X.transpose(1, 2))                      # (B,3,3)
            reg_plane = (G - I).pow(2).sum(dim=(1, 2)).mean() * self.wp

        # quats (use vector part only)
        reg_rot = torch.zeros((), device=device)
        if quats.numel():
            v = quats[..., 1:4]                                         # (B,Kq,3)
            v = v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            X = v.transpose(1, 2)                                       # (B,3,Kq)
            G = torch.matmul(X, X.transpose(1, 2))                      # (B,3,3)
            reg_rot = (G - I).pow(2).sum(dim=(1, 2)).mean() * self.wr

        return {"loss_reg_plane": reg_plane, "loss_reg_rot": reg_rot}
