#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import numpy as np
import torch

# 你的现有工具与模型
from src.SNC_dataloader import create_dataloader, load_config
from src.model.PrsNet_model import PRSNet
from src.model.Prs_Loss import SymmetryLoss, RegularizationLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_str: str) -> torch.device:
    if device_str and device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_ckpt_path(cfg) -> str:
    """
    优先从 cfg.testing.checkpoint 读取；若缺失，尝试 cfg.output.checkpoint；
    都没有则报错。
    """
    ckpt = None
    if hasattr(cfg, "testing") and hasattr(cfg.testing, "checkpoint"):
        ckpt = cfg.testing.checkpoint

    if ckpt is None:
        raise ValueError(
            "Checkpoint path not found. Please add 'testing.checkpoint' in config.yaml, e.g.\n"
            "testing:\n  checkpoint: ./checkpoints/<run_ts>/<epoch>_net_PRSNet.pth"
        )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
    return ckpt


def build_model_from_cfg(cfg, device: torch.device) -> PRSNet:
    # 与训练一致的构建
    model = PRSNet(
        input_nc=cfg.model.input_nc,
        base_channels=cfg.model.output_nc,
        conv_layers=cfg.model.conv_layers,
        num_planes=cfg.model.num_plane,
        num_quats=cfg.model.num_quat,
        use_bn=bool(cfg.model.use_bn),
        activation=cfg.model.activation,
        dropout=0.0,
    ).to(device)
    return model


def build_losses_from_cfg(cfg) -> tuple[SymmetryLoss, RegularizationLoss, float]:
    """
    返回：sym_loss, reg_loss, w_sym_reg
    """
    sym_loss = SymmetryLoss(
        grid_size=cfg.loss.grid_size,
        grid_bound=cfg.loss.grid_bound,
        ref_weight=cfg.loss.ref_weight,
        rot_weight=cfg.loss.rot_weight,
    )
    reg_loss = RegularizationLoss(
        weight_plane=cfg.loss.reg_plane_weight,
        weight_rot=cfg.loss.reg_rot_weight,
    )
    return sym_loss, reg_loss, float(cfg.loss.w_sym_reg)


def main():
    # 1) 读取配置与设备
    cfg = load_config("config.yaml")
    set_seed(cfg.training.seed)
    device = select_device(cfg.training.device)

    # 2) DataLoader：使用 test split
    #    你的 create_dataloader 内部会按 split='test' 使用 cfg.dataset.test_split 目录
    test_loader = create_dataloader(cfg, split="test")
    if len(test_loader.dataset) == 0:
        print("No test samples found.")
        return

    # 3) 构建模型与损失
    model = build_model_from_cfg(cfg, device)
    sym_loss, reg_loss, w_sym_reg = build_losses_from_cfg(cfg)

    # 4) 加载权重（从 config.yaml 指定的 checkpoint）
    ckpt_path = get_ckpt_path(cfg)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_ok = False
    try:
        model.load_state_dict(state, strict=True)
        load_ok = True
    except Exception as e:
        # 有些权重保存时包了一层 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            try:
                model.load_state_dict(state["state_dict"], strict=True)
                load_ok = True
            except Exception as e2:
                print(f"[WARN] strict load from state['state_dict'] failed: {e2}")
        if not load_ok:
            # 宽松加载（仅加载匹配的层）
            model.load_state_dict(state, strict=False)
            print("[WARN] Loaded with strict=False (some keys mismatched).")

    model.eval()

    # 5) 评测循环（无梯度）
    # 采用“按样本计权的平均值”：batch loss * batch_size，最后除以总样本数
    totals = {
        "ref": 0.0,
        "rot": 0.0,
        "regp": 0.0,
        "regr": 0.0,
        "total": 0.0,
    }
    n_samples = 0

    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            voxel  = batch["voxel"].to(device)    # (B,1,gs,gs,gs)
            points = batch["sample"].to(device)   # (B,P,3)
            cp     = batch["cp"].to(device)       # (B, gs^3, 3)

            quats, planes = model(voxel)          # quats:(B,Kq,4), planes:(B,Kp,4)

            losses_sym = sym_loss(points, cp, voxel, planes, quats)
            losses_reg = reg_loss(planes, quats)

            # 组合总损失（与训练保持一致：reg 分量可再乘 w_sym_reg）
            total_loss = (
                losses_sym["loss_ref"]
                + losses_sym["loss_rot"]
                + w_sym_reg * (losses_reg["loss_reg_plane"] + losses_reg["loss_reg_rot"])
            )

            B = voxel.size(0)
            n_samples += B

            # 累加（按 batch 大小计权）
            totals["ref"]   += losses_sym["loss_ref"].item() * B
            totals["rot"]   += losses_sym["loss_rot"].item() * B
            totals["regp"]  += losses_reg["loss_reg_plane"].item() * B
            totals["regr"]  += losses_reg["loss_reg_rot"].item() * B
            totals["total"] += total_loss.item() * B

    # 6) 输出全数据集平均损失
    if n_samples == 0:
        print("No valid samples in test set.")
        return

    avg = {k: v / n_samples for k, v in totals.items()}
    elapsed = time.time() - start_time

    print("\n===== Test Summary =====")
    print(f"Samples: {n_samples}")
    print(f"Loss_ref:        {avg['ref']:.6f}")
    print(f"Loss_rot:        {avg['rot']:.6f}")
    print(f"Loss_reg_plane:  {avg['regp']:.6f}")
    print(f"Loss_reg_rot:    {avg['regr']:.6f}")
    print(f"Total loss:      {avg['total']:.6f}")
    print(f"Elapsed:         {elapsed:.2f}s")


if __name__ == "__main__":
    main()
