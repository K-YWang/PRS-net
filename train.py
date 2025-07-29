import os
import random
import numpy as np
import torch

from src.SNC_dataloader import create_dataloader , load_config
from src.model.PrsNet_model import PRSNet
from src.model.Prs_Loss import SymmetryLoss, RegularizationLoss

from src.utils.method import init_logger, find_latest_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_str: str) -> torch.device:
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    cfg = load_config("config.yaml")
    set_seed(cfg.training.seed)

    device = select_device(cfg.training.device)

    os.makedirs(cfg.output.save_dir, exist_ok=True)
    os.makedirs(cfg.output.log_dir, exist_ok=True)

    logger = init_logger(cfg.output.log_dir)

    # DataLoader
    train_loader = create_dataloader(cfg, split=cfg.dataset.train_split)

    # Model
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

    # Losses
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

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
    )

    # 恢复训练
    start_epoch = 0
    model_ckpt, optim_ckpt, last_epoch = find_latest_checkpoint(cfg.output.save_dir)
    if model_ckpt is not None:
        try:
            state = torch.load(model_ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
            logger.info(f"Loaded model checkpoint: {model_ckpt}")

            if os.path.isfile(optim_ckpt):
                opt_state = torch.load(optim_ckpt, map_location="cpu")
                optimizer.load_state_dict(opt_state)
                logger.info(f"Loaded optimizer state: {optim_ckpt}")
            else:
                logger.info("Optimizer state not found; resume with fresh optimizer.")

            start_epoch = last_epoch + 1
            logger.info(f"Resume training from epoch {start_epoch}/{cfg.training.epochs}")
        except Exception as e:
            logger.exception(f"Failed to load checkpoint: {e}")
            logger.info("Start training from scratch.")

    logger.info("================================ Training begin ================================")
    # Train
    model.train()
    for epoch in range(start_epoch, cfg.training.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")

        running = {"ref": 0.0, "rot": 0.0, "regp": 0.0, "regr": 0.0, "tot": 0.0}
        steps = 0

        for batch in train_loader:
            if batch is None:
                continue

            voxel = batch["voxel"].to(device)    # (B,1,gs,gs,gs)
            points = batch["sample"].to(device)  # (B,P,3)
            cp = batch["cp"].to(device)          # (B, gs^3, 3)

            quats, planes = model(voxel)         # quats:(B,Kq,4), planes:(B,Kp,4)

            losses_sym = sym_loss(points, cp, voxel, planes, quats)
            losses_reg = reg_loss(planes, quats)

            loss = (losses_sym["loss_ref"]
                    + losses_sym["loss_rot"]
                    + cfg.loss.w_sym_reg * (losses_reg["loss_reg_plane"]
                    + losses_reg["loss_reg_rot"]))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # 统计
            running["ref"]  += losses_sym["loss_ref"].item()
            running["rot"]  += losses_sym["loss_rot"].item()
            running["regp"] += losses_reg["loss_reg_plane"].item()
            running["regr"] += losses_reg["loss_reg_rot"].item()
            running["tot"]  += loss.item()
            steps += 1

        # 日志与保存
        avg = {k: v / max(1, steps) for k, v in running.items()}
        logger.info(f"[Epoch {epoch:03d}] "
                    f"ref={avg['ref']:.6f} rot={avg['rot']:.6f} "
                    f"regp={avg['regp']:.6f} regr={avg['regr']:.6f} "
                    f"total={avg['tot']:.6f}  steps={steps}")

        # 保存模型与优化器（便于完整恢复）
        model_path = os.path.join(cfg.output.save_dir, f"{epoch:05d}_net_PRSNet.pth")
        optim_path = os.path.join(cfg.output.save_dir, f"{epoch:05d}_optim.pth")
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)
        logger.info(f"Saved checkpoint: {model_path} & {optim_path}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
