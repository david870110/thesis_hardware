from __future__ import annotations
import math
import os
import json
import numpy as np
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
import torch.ao.nn.qat as nnqat
from torch.nn.modules.utils import _pair

# -----------------------------------------------------------------------------
# 公用小工具
# -----------------------------------------------------------------------------

def _pair_int(v):
    if isinstance(v, tuple):
        return (int(v[0]), int(v[1]))
    v = int(v)
    return (v, v)


def choose_backend(name: str) -> str:
    name = (name or "fbgemm").lower()
    if name not in ("fbgemm", "qnnpack"):
        name = "fbgemm"
    torch.backends.quantized.engine = name
    return name


def make_qconfig(mode: str, backend: str, per_channel: bool) -> tq.QConfig:
    if mode == "qat":
        base = tq.get_default_qat_qconfig(backend)
        if per_channel:
            w = tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
            return tq.QConfig(activation=base.activation, weight=w)
        return base
    # PTQ
    act = tq.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    w = tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric) if per_channel \
        else tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    return tq.QConfig(activation=act, weight=w)


# -----------------------------------------------------------------------------
# 二值卷積/激活（維持原接口）
# -----------------------------------------------------------------------------

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sign()
    @staticmethod
    def backward(ctx, g: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).float()

def sign(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class BinaryAct(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sign(x)
        return torch.where(y == 0, torch.ones_like(y), y)


class BinaryConv2d(nn.Module):
    """不繼承 nn.Conv2d；以 F.conv2d 明確傳參，避免底層型別不匹配。"""
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1, bias=False, groups=1, dilation=1, use_alpha=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch // groups, k, k))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = _pair_int(stride)
        self.padding = _pair_int(padding)
        self.dilation = _pair_int(dilation)
        self.groups = int(groups)
        self.use_alpha = use_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.use_alpha:
            alpha = w.detach().abs().mean(dim=(1, 2, 3), keepdim=True)
            wq = (sign(w) * alpha).clamp(-1, 1)
        else:
            wq = sign(w)
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)


# -----------------------------------------------------------------------------
# 自寫普通卷積：PlainConv2dAlg（不用 nn.Conv2d/F.conv2d）
# -----------------------------------------------------------------------------

class PlainConv2dAlg(nn.Module):
    """im2col(F.unfold) + 分組矩陣乘法 + reshape 的純 Python 參考實作。
    支援 stride/padding/dilation/groups；若輸入是量化張量會先 dequantize。
    權重/偏置與 nn.Conv2d 同 shape，初始化一致。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        assert self.in_channels % self.groups == 0 and self.out_channels % self.groups == 0

        w_shape = (self.out_channels, self.in_channels // self.groups, *self.kernel_size)
        self.weight = nn.Parameter(torch.empty(*w_shape))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _out_hw(self, H, W):
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return Ho, Wo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(x, "is_quantized") and x.is_quantized:
            x = x.dequantize()
        N, Cin, H, W = x.shape
        G = self.groups
        Cin_g = Cin // G
        Cout_g = self.out_channels // G
        kh, kw = self.kernel_size
        Ho, Wo = self._out_hw(H, W)

        cols = F.unfold(x, kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=self.stride)
        L = cols.shape[-1]
        cols_g = cols.view(N, G, Cin_g * kh * kw, L)
        Wmat = self.weight.view(G, Cout_g, Cin_g * kh * kw)
        out_g = torch.einsum("gok,ngkl->ngol", Wmat, cols_g)
        out = out_g.reshape(N, self.out_channels, L).view(N, self.out_channels, Ho, Wo)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# -----------------------------------------------------------------------------
# SCBipolarConv2d（保持你原本行為；僅整理排版/註解）
# -----------------------------------------------------------------------------
# -------------------------- helpers --------------------------

def _ensure_dir(d: str):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _to_np(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bool:
            return x.to(torch.uint8).numpy()
        if x.dtype in (torch.float64,):
            return x.to(torch.float32).numpy()
        if x.dtype in (torch.int64,):
            # use int32 when safe to save space
            if x.numel() == 0:
                return x.numpy()
            a = x.abs().max().item()
            if a <= (2**31 - 1):
                return x.to(torch.int32).numpy()
        return x.numpy()
    return np.array(x)

def _save_npz(path: str, **arrays):
    # convert tensors to numpy first
    tosave = {}
    for k, v in arrays.items():
        if isinstance(v, dict):
            # flatten one level: k__sub
            sub = {f"{k}__{kk}": vv for kk, vv in v.items()}
            for kk, vv in sub.items():
                tosave[kk] = _to_np(vv)
        else:
            tosave[k] = _to_np(v)
    np.savez(path, **tosave)

class _GoldenDumper:
    def __init__(self, base_dir: Optional[str], tag: Optional[str] = None, enable: bool = False):
        self.base_dir = base_dir
        self.tag = tag
        self.enable = bool(enable) and bool(base_dir)
        self.step = 0
        if self.enable:
            _ensure_dir(base_dir)

    def fname(self, stem: str) -> str:
        tag = (self.tag + "_") if self.tag else ""
        return os.path.join(self.base_dir, f"S{self.step:02d}_{tag}{stem}.npz")

    def dump(self, stem: str, **arrays):
        if not self.enable:
            self.step += 1
            return
        path = self.fname(stem)
        _save_npz(path, **arrays)
        self.step += 1

# -------------------------- ported utilities --------------------------
# -------------------------- ported utilities --------------------------

def _pair_int(v):
    if isinstance(v, tuple):
        return (int(v[0]), int(v[1]))
    v = int(v)
    return (v, v)

def _compute_out_hw(H: int, W: int, kernel_size, stride, padding, dilation):
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
    pH, pW = padding if isinstance(padding, tuple) else (padding, padding)
    dH, dW = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    Hout = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    Wout = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return Hout, Wout

def _clamp_int8(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -128, 127)

def quantize_int8_from_float(x: torch.Tensor, scale: float) -> torch.Tensor:
    if x.dtype == torch.int8:
        return x
    q = torch.round(x * scale)
    return _clamp_int8(q).to(torch.int8)

def _quota_from_qint(q: torch.Tensor, T: int) -> torch.Tensor:
    u = (q.to(torch.int16) + 128).to(torch.int32)   # 0..255
    s = (u * int(T) + 128) >> 8                    # round((u/255)*T)
    return s

# -------------------------- operator --------------------------

class SCBipolarConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bitstream_length: int = 64,
                 stride_x: int = 17, stride_w: int = 53,
                 phase_x_base: int = 0, phase_w_base: int = 0,
                 phase_style: str = "nlk_outk",  # "const" or "nlk_outk"
                 input_scale: float = 128.0,     # legacy symmetric quant scale
                 weight_perchannel: bool = True,
                 return_int8: bool = False, out_scale: int = 128,
                 # --- golden logging toggles ---
                 debug_dump_dir: Optional[str] = None,
                 debug_dump_bits: bool = True,
                 debug_dump_eq: bool = False,
                 debug_tag: Optional[str] = None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if groups != 1:
            raise NotImplementedError("SCBipolarConv2d only supports groups==1 for now")

        # SC params
        self.T = int(bitstream_length)
        self.stride_x = int(stride_x)
        self.stride_w = int(stride_w)
        self.phase_x_base = int(phase_x_base)
        self.phase_w_base = int(phase_w_base)
        self.phase_style = str(phase_style)

        # quant/output
        self.input_scale = float(input_scale)
        self.weight_perchannel = bool(weight_perchannel)
        self.return_int8 = bool(return_int8)
        self.out_scale = int(out_scale)
        self._eps = 1e-12

        # observer qparams (optional affine input)
        self.use_observer_qparams = False
        self.act_scale = None
        self.act_zp = None
        self.act_qmin = 0
        self.act_qmax = 255
        self._dbg_qparams_printed = False

        # logging
        self._gd = _GoldenDumper(debug_dump_dir, tag=debug_tag, enable=(debug_dump_dir is not None))
        self._dbg_dump_bits = bool(debug_dump_bits)
        self._dbg_dump_eq = bool(debug_dump_eq)

    # externally set activation observer params
    def set_input_qparams(self, scale: float, zp: int, qmin: int = 0, qmax: int = 255):
        self.act_scale = float(scale)
        self.act_zp = int(zp)
        self.act_qmin = int(qmin)
        self.act_qmax = int(qmax)
        self.use_observer_qparams = True

    def _add_bias_safe(self, y: torch.Tensor) -> torch.Tensor:
        b = getattr(self, "bias", None)
        if b is None:
            return y
        b = b.to(device=y.device, dtype=y.dtype)
        if b.ndim == 1 and b.numel() == y.size(1):
            b = b.view(1, -1, 1, 1)
        else:
            b = b.view(1, 1, 1, 1)
        return y + b

    @staticmethod
    def _im2col(x: torch.Tensor, kernel_size, stride, padding, dilation) -> torch.Tensor:
        x_f = x.float() if x.dtype != torch.float32 else x
        return F.unfold(x_f, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    def _quantize_input_cols(self, cols_f: torch.Tensor, input_scale: float, orig_dtype: torch.dtype) -> torch.Tensor:
        if getattr(self, "use_observer_qparams", False) and (self.act_scale is not None):
            q = torch.round(cols_f / self.act_scale) + self.act_zp
            q = torch.clamp(q, self.act_qmin, self.act_qmax)
            rng = float(self.act_qmax - self.act_qmin)
            if rng <= 0:
                q255 = torch.zeros_like(q)
            else:
                q255 = torch.round((q - self.act_qmin) * (255.0 / rng))
            q255 = torch.clamp(q255, 0, 255)
            q_int8 = (q255.to(torch.int16) - 128).clamp_(-128, 127).to(torch.int8)
            if not self._dbg_qparams_printed:
                print(f"[SCConv] use_observer_qparams=True  scale={self.act_scale:.6g} "
                      f"zp={self.act_zp} qmin={self.act_qmin} qmax={self.act_qmax} (rescale→0..255)")
                self._dbg_qparams_printed = True
            return q_int8

        if orig_dtype == torch.int8:
            return cols_f.round().to(torch.int8)
        return quantize_int8_from_float(cols_f, input_scale)

    def _quantize_weight_int8(self) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.weight
        if self.weight_perchannel:
            wv = w.view(self.out_channels, -1)
            maxabs = torch.amax(torch.abs(wv), dim=1).clamp_min(self._eps)  # [Cout]
            scale = (128.0 / maxabs).view(self.out_channels, 1, 1, 1)
            qw = torch.round(w * scale).clamp_(-128, 127).to(torch.int8)
            return qw, maxabs
        else:
            maxabs = torch.amax(torch.abs(w)).clamp_min(self._eps)
            scale = 128.0 / maxabs
            qw = torch.round(w * scale).clamp_(-128, 127).to(torch.int8)
            return qw, maxabs

    def _packbits_last(self, bool_tensor: torch.Tensor) -> Dict[str, Any]:
        """Pack boolean tensor along the last axis to reduce size."""
        arr = bool_tensor.detach().cpu().numpy().astype(np.uint8)  # 0/1
        packed = np.packbits(arr, axis=-1)
        return {"shape": np.array(arr.shape, dtype=np.int32), "packed": packed}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        N, Cin, H, W = x.shape
        T = self.T
        if Cin != self.in_channels:
            raise RuntimeError(f"SCBipolarConv2d: expected {self.in_channels} in_channels, got {Cin}")

        # --- Step 00: meta + input ---
        meta = {
            "T": int(T), "stride_x": int(self.stride_x), "stride_w": int(self.stride_w),
            "phase_x_base": int(self.phase_x_base), "phase_w_base": int(self.phase_w_base),
            "phase_style": str(self.phase_style),
            "kernel_size": tuple(_pair_int(self.kernel_size)),
            "stride": tuple(_pair_int(self.stride)),
            "padding": tuple(_pair_int(self.padding)),
            "dilation": tuple(_pair_int(self.dilation)),
            "in_shape": (int(N), int(Cin), int(H), int(W)),
            "return_int8": bool(self.return_int8), "out_scale": int(self.out_scale),
            "use_observer_qparams": bool(self.use_observer_qparams),
            "act_scale": float(self.act_scale) if self.act_scale is not None else None,
            "act_zp": int(self.act_zp) if self.act_zp is not None else None,
            "act_qmin": int(self.act_qmin), "act_qmax": int(self.act_qmax),
        }
        # self._gd.dump("meta_and_input", meta=meta, x=x)

        
        # --- Step 01: unfold(im2col) ---
        x_cols_f = self._im2col(x, self.kernel_size, self.stride, self.padding, self.dilation)
        # self._gd.dump("im2col_x_cols_f", x_cols_f=x_cols_f)

        # --- Step 02: quantize input cols -> int8 ---
        x_cols_q = self._quantize_input_cols(x_cols_f, self.input_scale, x.dtype).to(device)  # int8
        # self._gd.dump("x_cols_q_int8", x_cols_q=x_cols_q)

        CK = x_cols_q.shape[1]
        H_out, W_out = _compute_out_hw(H, W, self.kernel_size, self.stride, self.padding, self.dilation)
        L = H_out * W_out

        # --- Step 03: quantize weights to int8 + per-channel maxabs ---
        qw, maxabs = self._quantize_weight_int8()
        qw = qw.to(device)                                           # [Cout, Cin, kH, kW]
        w_cols_q = qw.view(self.out_channels, CK)                    # [Cout, CK]
        # self._gd.dump("weight_quant", w_float=self.weight, w_int8=qw, w_cols_q=w_cols_q, maxabs=maxabs)

        # --- Step 04: reshape x to [NL, CK] ---
        x_mat_q = x_cols_q.permute(0, 2, 1).contiguous().view(N * L, CK)  # [NL, CK]
        # self._gd.dump("x_mat_q_NLxCK", x_mat_q=x_mat_q)

        # --- Step 05: quota from qint ---
        s_x = _quota_from_qint(x_mat_q, T)            # [NL, CK]  int32
        s_w = _quota_from_qint(w_cols_q, T)           # [Cout, CK] int32
        # self._gd.dump("quota_sx_sw", s_x=s_x, s_w=s_w)

        # --- Step 06: phase bases ---
        nl = N * L
        # l_idx = torch.arange(L, device=device, dtype=torch.int64).repeat(N).view(nl, 1)
        k_idx = torch.arange(CK, device=device, dtype=torch.int64).view(1, CK) % 4
        kw_idx = torch.zeros(CK, device=device, dtype=torch.int64).view(1, CK)

        base_x = (self.phase_x_base + k_idx) % T         # [NL, CK]
        # base_x = (self.phase_x_base + l_idx + k_idx) % T         # [NL, CK]
        # out_idx = torch.arange(self.out_channels, device=device, dtype=torch.int64).view(self.out_channels, 1)
        base_w = (self.phase_w_base + k_idx) % T               # [Cout, CK]

        # --- Step 07: validate strides ---
        stride_x = self.stride_x % T
        stride_w = self.stride_w % T
        if math.gcd(stride_x, T) != 1:
            raise ValueError(f"stride_x={self.stride_x} and T={T} are not coprime")
        if math.gcd(stride_w, T) != 1:
            raise ValueError(f"stride_w={self.stride_w} and T={T} are not coprime")
        # self._gd.dump("stride_mods", stride_x=stride_x, stride_w=stride_w, T=T)

        # --- Step 08k : per-k accumulation (等價於逐 t；更硬體友善) ---
        prod_bxbw = torch.zeros((nl, self.out_channels), dtype=torch.float32, device=device)

        # 事先把 t 的位移表建好（Weyl 步進），避免在迴圈內乘法
        t_off_x = (torch.arange(T, device=device, dtype=torch.int64) * stride_x) % T  # [T]
        t_off_w = (torch.arange(T, device=device, dtype=torch.int64) * stride_w) % T  # [T]


        # 以 k 為外迴圈，分塊處理避免中間張量過大
        k_tile = 32  # 可依你的顯存情況調整（16/32/64/128…）

        for ks in range(0, CK, k_tile):
            ke = min(ks + k_tile, CK)
            Ks = ke - ks

            bx = base_x[:, ks:ke]           # [NL, Ks]
            bw = base_w[:, ks:ke]           # [Cout, Ks]
            sx_ = s_x[:, ks:ke]             # [NL, Ks]
            sw_ = s_w[:, ks:ke]             # [Cout, Ks]

            # 生成整段 bitstream（沿 t 維度），一次比對整段再 popcount
            # idx_x: [NL, Ks, T]；bit_x: [NL, Ks, T]
            idx_x = (bx.unsqueeze(-1) + t_off_x) % T
            bit_x = (idx_x < sx_.unsqueeze(-1))

            # idx_w: [Cout, Ks, T]；bit_w: [Cout, Ks, T]
            idx_w = (bw.unsqueeze(-1) + t_off_w) % T
            bit_w = (idx_w < sw_.unsqueeze(-1))

            # eq over T for each k：s_eq_k ∈ [0..T]
            # s_eq_k: [NL, Cout, Ks]
            s_eq_k = (bit_x.unsqueeze(1) == bit_w.unsqueeze(0)).sum(dim=-1)

            # 將每個 k 的「2*s_eq_k - T」累加，最後再除以 T（和逐 t 完全等價）
            sscb = (2.0 * s_eq_k.float() - T)
            # 這裡先對 Ks 加總得到 [NL, Cout]，再加到 prod_bxbw
            prod_bxbw += ( (sscb.sum(dim=2)) / float(T) )



        # --- Step 09: decode coefficients ---
        if self.use_observer_qparams and (self.act_scale is not None):
            rng = float(self.act_qmax - self.act_qmin)
            A_x = self.act_scale * (128.0 * rng / 255.0)                     # scalar
            B_x = self.act_scale * (self.act_qmin - self.act_zp + 128.0 * rng / 255.0)  # scalar
        else:
            A_x = 128.0 / self.input_scale
            B_x = 0.0

        if self.weight_perchannel:
            A_w = maxabs.to(device).view(1, -1)    # [1, Cout]
        else:
            A_w = maxabs.to(device).view(1, 1)     # scalar
        B_w = 0.0

        sum_bw = (2.0 * (s_w.sum(dim=1).float() / float(T)) - float(CK)).view(1, -1)  # [1, Cout]

        # --- Step 10: reconstruct y_cols ---
        # y_cols = (A_x * A_w) * prod_bxbw + (B_x * A_w) * sum_bw        # [NL, Cout]
        y_cols = prod_bxbw 


        # --- Step 11: fold back to feature map ---S
        y = y_cols.view(N, L, self.out_channels).permute(0, 2, 1).contiguous()
        y = y.view(N, self.out_channels, H_out, W_out)
        # self._gd.dump("y_feature_before_bias", y=y)

        # --- Step 12: add bias ---
        y = self._add_bias_safe(y)
        # self._gd.dump("y_after_bias", y=y)

        # --- Step 13: (optional) int8 return ---
        if self.return_int8:
            y_int8 = torch.clamp(torch.round(y * float(self.out_scale)), -128, 127).to(torch.int8)
            # self._gd.dump("y_int8", y_int8=y_int8)
            return y_int8.to(torch.float32)

        return y
# -----------------------------------------------------------------------------
# VGG16 組裝 + QAT/量化工具
# -----------------------------------------------------------------------------

# def make_vgg16_features(binarize_from=1, sc_first=True, sc_T=1) -> nn.Sequential:
#     cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
#     layers, in_ch, ci = [], 3, 0
#     for v in cfg:
#         if v == 'M':
#             layers.append(nn.MaxPool2d(2, 2)); continue
#         if ci == 0 and sc_first:
#             layers += [SCBipolarConv2d(in_ch, v, 3, 1, 1, bias=False, bitstream_length=sc_T, return_int8=False,debug_dump_dir="./golden/run1",
#             debug_dump_bits=True,debug_dump_eq=False,debug_tag="batch0"),nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#         elif ci >= binarize_from:
#             layers += [BinaryConv2d(in_ch, v, 3, 1, 1, False), nn.BatchNorm2d(v), BinaryAct()]
#         else:
#             layers += [nn.Conv2d(in_ch, v, 3, 1, 1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#         in_ch, ci = v, ci + 1
#     return nn.Sequential(*layers)

def make_vgg16_features(binarize_from=1, sc_first=True, sc_T=1) -> nn.Sequential:
    cfg = [64,64,'M',128,128,'M',
           256,256,256,'M',
           512,512,512,'M',
           512,512,512,'M']
    layers = []
    in_ch = 3
    ci = 0  # 第幾個 conv（不算 MaxPool）

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
            continue

        # 前面純實數區：Conv + BN + ReLU
        if ci < binarize_from - 1:
            layers += [
                nn.Conv2d(in_ch, v, 3, 1, 1, bias=False),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ]

        # 最後一層實數 Conv：只做 Conv + BN，不做 ReLU
        elif ci == binarize_from - 1:
            layers += [
                nn.Conv2d(in_ch, v, 3, 1, 1, bias=False),
                nn.BatchNorm2d(v),
            ]

        # 第一個 Binary block：BinaryAct + BinaryConv2d + BN + BinaryAct
        elif ci == binarize_from:
            layers += [
                BinaryAct(),                                  # ★ sign(BN 輸出，有正有負)
                BinaryConv2d(in_ch, v, 3, 1, 1, False),
                nn.BatchNorm2d(v),
                BinaryAct(),
            ]

        # 後面的 Binary block：BinaryConv2d + BN + BinaryAct
        else:  # ci > binarize_from
            layers += [
                BinaryConv2d(in_ch, v, 3, 1, 1, False),
                nn.BatchNorm2d(v),
                BinaryAct(),
            ]

        in_ch = v
        ci += 1

    return nn.Sequential(*layers)


class BinaryVGG16(nn.Module):
    def __init__(self, num_classes=10, binarize_from=1, dropout=0.0):
        super().__init__()
        self.features = make_vgg16_features(binarize_from)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = (nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512, num_classes))
                           if dropout > 0 else nn.Linear(512, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x); x = self.pool(x); x = torch.flatten(x, 1)
        return self.classifier(x)


class BNNQuantWrapper(nn.Module):
    """只量化 head/cls，保留中段二值區塊不量化。"""
    def __init__(self, bnn: BinaryVGG16):
        super().__init__()
        first_bin = None
        for i, m in enumerate(bnn.features):
            if isinstance(m, BinaryConv2d):
                first_bin = i; break
        if first_bin is None:
            first_bin = len(bnn.features)
        self.head = nn.Sequential(*[bnn.features[i] for i in range(0, first_bin)])
        self.tail = nn.Sequential(*[bnn.features[i] for i in range(first_bin, len(bnn.features))])
        self.pool = bnn.pool
        self.cls = bnn.classifier
        self.q_in, self.dq_head = tq.QuantStub(), tq.DeQuantStub()
        self.q_cls, self.dq_out = tq.QuantStub(), tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_in(x); x = self.head(x); x = self.dq_head(x)
        x = self.tail(x); x = self.pool(x); x = torch.flatten(x, 1)
        x = self.q_cls(x); x = self.cls(x); return self.dq_out(x)


def fuse_head(seq: nn.Sequential) -> None:
    changed = True
    while changed:
        changed = False
        mods = list(seq.named_children())
        i = 0
        while i < len(mods):
            name, m = mods[i]
            if isinstance(m, SCBipolarConv2d):  # SC 不支援 fuser
                i += 1; continue
            if isinstance(m, nn.Conv2d):
                if i + 2 < len(mods) and isinstance(mods[i+1][1], nn.BatchNorm2d) and isinstance(mods[i+2][1], nn.ReLU):
                    tq.fuse_modules(seq, [mods[i][0], mods[i+1][0], mods[i+2][0]], inplace=True)
                    changed = True; break
                if i + 1 < len(mods) and isinstance(mods[i+1][1], nn.BatchNorm2d):
                    tq.fuse_modules(seq, [mods[i][0], mods[i+1][0]], inplace=True)
                    changed = True; break
                if i + 1 < len(mods) and isinstance(mods[i+1][1], nn.ReLU):
                    tq.fuse_modules(seq, [mods[i][0], mods[i+1][0]], inplace=True)
                    changed = True; break
            i += 1


def _clear_qconfig(model: nn.Module) -> None:
    for m in model.modules():
        m.qconfig = None


def _force_qat_modules(wrapped: BNNQuantWrapper, qcfg: tq.QConfig) -> None:
    for name, mod in list(wrapped.head.named_children()):
        if isinstance(mod, SCBipolarConv2d):
            mod.qconfig = None
            continue
        if isinstance(mod, nn.Conv2d):
            mod.qconfig = qcfg
            setattr(wrapped.head, name, nnqat.Conv2d.from_float(mod))
    if isinstance(wrapped.cls, nn.Linear):
        wrapped.cls.qconfig = qcfg
        wrapped.cls = nnqat.Linear.from_float(wrapped.cls)
    elif isinstance(wrapped.cls, nn.Sequential):
        for n, mod in list(wrapped.cls.named_children()):
            if isinstance(mod, nn.Linear):
                mod.qconfig = qcfg
                setattr(wrapped.cls, n, nnqat.Linear.from_float(mod))
                break


def prepare_int8_model(float_model: nn.Module, args) -> nn.Module:
    backend = choose_backend(getattr(args, "quant_backend", "fbgemm"))
    qcfg = make_qconfig(getattr(args, "int8", "qat"), backend, getattr(args, "per_channel", False))

    wrapped = BNNQuantWrapper(float_model)
    wrapped.eval(); fuse_head(wrapped.head)

    _clear_qconfig(wrapped)
    for m in wrapped.head.modules():
        if isinstance(m, SCBipolarConv2d):
            m.qconfig = None
        elif isinstance(m, (nn.Conv2d, nn.ReLU, nn.ReLU6, nn.Sequential)):
            m.qconfig = qcfg
    for m in wrapped.cls.modules():
        if isinstance(m, (nn.Linear, nn.ReLU, nn.ReLU6, nn.Sequential)):
            m.qconfig = qcfg
    wrapped.q_in.qconfig = qcfg
    wrapped.q_cls.qconfig = qcfg
    tq.propagate_qconfig_(wrapped)

    _force_qat_modules(wrapped, qcfg)

    if getattr(args, "int8", "qat") == "qat":
        wrapped.train(); tq.prepare_qat(wrapped, inplace=True)
    else:
        tq.prepare(wrapped, inplace=True); wrapped.eval()
    return wrapped


def convert_to_int8(prepared: nn.Module) -> nn.Module:
    return tq.convert(prepared.eval().cpu(), inplace=False)

# -----------------------------------------------------------------------------
# 與舊版 bnn_models.py 保持介面的保險函式：_sanitize_conv_hparams
# -----------------------------------------------------------------------------
def _sanitize_conv_hparams(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            s = _pair_int(getattr(m, "stride", (1, 1)))
            p = _pair_int(getattr(m, "padding", (0, 0)))
            d = _pair_int(getattr(m, "dilation", (1, 1)))
            g = int(getattr(m, "groups", 1))
            s = (max(1, s[0]), max(1, s[1]))
            p = (max(0, p[0]), max(0, p[1]))
            d = (max(1, d[0]), max(1, d[1]))
            if g <= 0 or getattr(m, "in_channels", 1) % g != 0:
                g = 1
            m.stride, m.padding, m.dilation, m.groups = s, p, d, g
