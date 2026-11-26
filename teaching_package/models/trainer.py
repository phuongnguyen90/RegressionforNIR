"""Huấn luyện tự giám sát SpecMultiMAE-AD trên phổ nền (Section 5.3).

AdamW (lr 1e-4, weight decay 0.05), lịch cosine với khởi động tuyến tính 10 epoch,
dừng sớm theo MSE tái tạo trên tập kiểm định phổ nền (kiên nhẫn 50 epoch).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from masking.cmsm import batch_masks, scattered_patch_mask
from .casrl import casrl_loss
from .specmultimae import build_model


def _masks(B, N, cfg, rng, device):
    if cfg["masking"].get("strategy", "cmsm") == "scattered":
        return scattered_patch_mask(B, N, cfg["masking"]["mask_ratio"], rng, device)
    return batch_masks(B, N, cfg["masking"], rng, device)


@torch.no_grad()
def val_mse(model, X, wl, cfg, rng, device, bs=256):
    model.eval()
    P = cfg["model"]["P"]
    N = math.ceil(X.shape[1] / P)
    tot, cnt = 0.0, 0
    for i in range(0, len(X), bs):
        x = X[i : i + bs]
        m, _ = _masks(len(x), N, cfg, rng, device)
        xh = model(x, wl, m)
        wm = m.repeat_interleave(P, 1)[:, : X.shape[1]].float()
        tot += float((((x - xh) ** 2) * wm).sum())
        cnt += float(wm.sum())
    return tot / max(cnt, 1)


def train(
    X_tr: np.ndarray,
    X_val: np.ndarray,
    wl: np.ndarray,
    cfg: dict,
    device="cpu",
    seed: int = 0,
    log=print,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tc = cfg["train"]
    model = build_model(cfg).to(device)
    Xtr = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    Xva = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    wlt = torch.as_tensor(wl, dtype=torch.float32, device=device)
    P = cfg["model"]["P"]
    N = math.ceil(Xtr.shape[1] / P)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    steps_per_epoch = max(1, math.ceil(len(Xtr) / tc["batch_size"]))
    total, warm = (
        tc["epochs"] * steps_per_epoch,
        tc["warmup_epochs"] * steps_per_epoch,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / max(1, warm)
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
        ),
    )
    lc = cfg["loss"]
    best, best_state, bad = float("inf"), None, 0
    history = []
    t0 = time.time()
    for ep in range(tc["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        run = 0.0
        for i in range(0, len(Xtr), tc["batch_size"]):
            x = Xtr[perm[i : i + tc["batch_size"]]]
            m, segs = _masks(len(x), N, cfg, rng, device)
            xh = model(x, wlt, m)
            loss = casrl_loss(
                x, xh, m, segs, P, lc["lambda"], lc["beta"], lc.get("vanilla", False)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), tc.get("grad_clip", 1.0)
            )
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(x)
        v = val_mse(model, Xva, wlt, cfg, rng, device)
        history.append({"epoch": ep + 1, "train_loss": run / len(Xtr), "val_mse": v})
        if v < best - 1e-7:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if (ep + 1) % tc.get("log_every", 10) == 0 or ep == 0:
            log(
                f"epoch {ep + 1:4d}  loss {run / len(Xtr):.5f}  val_mse {v:.6f}  "
                f"best {best:.6f}  {time.time() - t0:.0f}s"
            )
        if bad >= tc["patience"]:
            log(f"dừng sớm ở epoch {ep + 1}")
            break
    model.load_state_dict(best_state)
    return model, history
