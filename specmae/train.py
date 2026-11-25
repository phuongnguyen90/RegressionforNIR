"""Self-supervised training on normal spectra.

AdamW (lr 1e-4, wd 0.05), linear warm-up then cosine annealing (per step),
mixed precision on CUDA, early stopping on validation masked-reconstruction MSE.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import LossConfig, TrainConfig
from .loss import compute_loss
from .masking import MaskSampler
from .model import SpecMultiMAE


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "mask_token" in name or "norm" in name or name.endswith("freqs"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def cosine_with_warmup(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    warmup_steps = max(1, warmup_steps)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.no_grad()
def masked_reconstruction_mse(model: SpecMultiMAE, X: np.ndarray, sampler: MaskSampler,
                              device: torch.device, seed: int = 12345, batch_size: int = 1024) -> float:
    """Masked-position MSE with a fixed mask stream (deterministic across epochs)."""
    model.eval()
    rng = np.random.default_rng(seed)
    num, den = 0.0, 0.0
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        mask_np, _ = sampler.sample(xb.shape[0], rng)
        mask = torch.from_numpy(mask_np).to(device)
        pred = model(xb, mask).float()
        pm = model.pixel_mask(mask).float()
        num += float((((pred - xb) ** 2) * pm).sum())
        den += float(pm.sum())
    return num / max(den, 1.0)


def train_model(model: SpecMultiMAE, sampler: MaskSampler, X_train: np.ndarray,
                X_val: Optional[np.ndarray], loss_cfg: LossConfig, cfg: TrainConfig,
                device: torch.device, seed: int = 0,
                log: Callable[[str], None] = print) -> List[Dict[str, float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)

    X_train = np.asarray(X_train, dtype=np.float32)
    if X_val is None or len(X_val) == 0:
        X_val = X_train[: min(len(X_train), 512)]
        log("[train] no validation normals; early stopping on a training subset")
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=min(cfg.batch_size, len(X_train)),
                        shuffle=True, generator=gen, num_workers=cfg.num_workers,
                        pin_memory=device.type == "cuda")
    steps_per_epoch = len(loader)
    total = cfg.epochs * steps_per_epoch
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(cfg.warmup_epochs * steps_per_epoch, total))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = _grad_scaler(use_amp)

    best, best_state, best_epoch, bad = math.inf, None, -1, 0
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        sums: Dict[str, float] = {}
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            mask_np, seg_np = sampler.sample(xb.shape[0], rng)
            mask = torch.from_numpy(mask_np).to(device)
            seg = model.pixel_segments(torch.from_numpy(seg_np).to(device))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(xb, mask)
            loss, parts = compute_loss(pred, xb, seg, loss_cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scale_before = scaler.get_scale() if use_amp else None
            scaler.step(opt)
            scaler.update()
            if not use_amp or scaler.get_scale() >= scale_before:  # AMP skips steps with inf grads
                sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v

        val = masked_reconstruction_mse(model, X_val, sampler, device, seed=seed + 1,
                                        batch_size=cfg.val_batch_size)
        rec = {"epoch": epoch, **{k: v / steps_per_epoch for k, v in sums.items()}, "val_mse": val,
               "lr": opt.param_groups[0]["lr"]}
        history.append(rec)
        if val < best:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if epoch % max(1, cfg.log_every) == 0 or epoch == cfg.epochs - 1:
            log(f"[train] epoch {epoch:4d} loss {rec['loss']:.5f} val_mse {val:.6f} "
                f"best {best:.6f}@{best_epoch} ({time.time() - t0:.0f}s)")
        if bad >= cfg.patience:
            log(f"[train] early stop at epoch {epoch}; best val_mse {best:.6f} at epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
