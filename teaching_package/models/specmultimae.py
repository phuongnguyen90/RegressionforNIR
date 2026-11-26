"""Bộ tự mã hóa mặt nạ Transformer bất đối xứng SpecMultiMAE-AD (Section 4.3, công
thức 18-23).

Luồng xử lý:
  phổ (B, D) -> mảnh (B, N, P) -> t_j = [e_j, w_j] (2d) -> chiếu sang d_e
  bộ mã hóa L_e khối Transformer chỉ nhận các mảnh nhìn thấy
  bộ giải mã L_d khối, nhận chuỗi đủ N token: z_j (chiếu sang d_d) hoặc token mặt nạ
  m_j (d_d),
      mỗi token ghép với nhúng bước sóng riêng của bộ giải mã w'_j (d_d) -> 2 d_d
  đầu chiếu tuyến tính 2 d_d -> P, ghép lại thành phổ tái tạo (B, D)

Ghi chú về kích thước (Section 5.3): d = 512, token sau ghép 1024; bộ mã hóa 12 lớp,
8 đầu, chiều ẩn 512; bộ giải mã 4 lớp, 8 đầu, chiều ẩn 256. Gói này hiểu "chiều ẩn"
là độ rộng của luồng dư trong khối Transformer, nên token 1024 chiều được chiếu
tuyến tính về 512 trước khi vào bộ mã hóa, và bộ giải mã dùng d_d = 128 cho mỗi nửa
token (2 d_d = 256). Các giá trị đều chỉnh được trong tệp cấu hình.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .embedding import (
    PatchEmbed,
    WavelengthEmbed,
    patch_centers,
    patchify,
    unpatchify,
)


def _blocks(width, depth, heads, mlp_ratio, dropout):
    layer = nn.TransformerEncoderLayer(
        width,
        heads,
        int(width * mlp_ratio),
        dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class SpecMultiMAE(nn.Module):
    def __init__(
        self,
        P=32,
        d=512,
        d_enc=512,
        enc_depth=12,
        enc_heads=8,
        d_dec=128,
        dec_depth=4,
        dec_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        wl_mode="fourier",
        n_freq=256,
        **_,
    ):
        super().__init__()
        self.P = P
        self.patch_embed = PatchEmbed(P, d)
        self.wl_enc = WavelengthEmbed(d, wl_mode, n_freq)
        self.enc_in = nn.Linear(2 * d, d_enc)
        self.encoder = _blocks(d_enc, enc_depth, enc_heads, mlp_ratio, dropout)
        self.enc_norm = nn.LayerNorm(d_enc)
        self.dec_in = nn.Linear(d_enc, d_dec)
        self.mask_token = nn.Parameter(torch.zeros(d_dec))
        nn.init.normal_(self.mask_token, std=0.02)
        self.wl_dec = WavelengthEmbed(
            d_dec, wl_mode, d_dec // 2 if wl_mode == "fourier" else None
        )
        self.decoder = _blocks(2 * d_dec, dec_depth, dec_heads, mlp_ratio, dropout)
        self.dec_norm = nn.LayerNorm(2 * d_dec)
        self.head = nn.Linear(2 * d_dec, P)

    def forward(self, x: torch.Tensor, wl: torch.Tensor, patch_mask: torch.Tensor):
        """x: (B, D) phổ đã tiền xử lý; wl: (D,) bước sóng (nm);
        patch_mask: bool (B, N), True = mảnh bị che. Mọi mẫu trong lô phải có cùng
        số mảnh bị che (CMSM bảo đảm điều này). Trả về phổ tái tạo đầy đủ (B, D)."""
        B, D = x.shape
        # S ⊙ (1 - m): xóa giá trị tại các bước sóng bị che (công thức 2)
        x = x * (~patch_mask).repeat_interleave(self.P, 1)[:, :D]
        patches = patchify(x, self.P)  # (B, N, P)
        N = patches.shape[1]
        lam = patch_centers(wl, self.P)  # bước sóng trung tâm mỗi mảnh, (N,)
        # t_j = [e_j, w_j] (công thức 9), rồi chiếu về độ rộng bộ mã hóa
        e = self.patch_embed(patches)
        w = self.wl_enc(lam).expand(B, N, -1)
        tok = self.enc_in(torch.cat([e, w], dim=-1))

        # bộ mã hóa chỉ nhận các mảnh nhìn thấy (công thức 20)
        n_vis = int((~patch_mask[0]).sum())
        vis_idx = torch.argsort(patch_mask.int(), dim=1, stable=True)[:, :n_vis]
        t_vis = torch.gather(tok, 1, vis_idx[..., None].expand(-1, -1, tok.shape[-1]))
        z_vis = self.enc_norm(self.encoder(t_vis))

        # bộ giải mã: đặt token đã mã hóa về vị trí cũ, chỗ trống là token mặt nạ,
        # mỗi token ghép với nhúng bước sóng của bộ giải mã (công thức 21-22)
        z = self.dec_in(z_vis)
        full = self.mask_token.expand(B, N, -1).clone()
        full.scatter_(1, vis_idx[..., None].expand(-1, -1, z.shape[-1]), z)
        full = torch.cat([full, self.wl_dec(lam).expand(B, N, -1)], dim=-1)
        out = self.head(self.dec_norm(self.decoder(full)))  # (B, N, P), công thức 23
        return unpatchify(out, D)


def build_model(cfg: dict) -> SpecMultiMAE:
    return SpecMultiMAE(**cfg["model"])
