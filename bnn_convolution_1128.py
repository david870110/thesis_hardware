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
        k_idx = torch.arange(CK, device=device, dtype=torch.int64) % 4
        # k_idx = torch.zeros(CK, device=device, dtype=torch.int64)
        nl = N * L
        if self.phase_style == "const":
            base_x = torch.full((nl, CK), int(self.phase_x_base % T), dtype=torch.int64, device=device)
            base_w = torch.full((self.out_channels, CK), int(self.phase_w_base % T), dtype=torch.int64, device=device)
        else:
            base_x = (self.phase_x_base + k_idx.view(1, CK)) % T         # [NL, CK]
            # out_idx = torch.arange(self.out_channels, device=device, dtype=torch.int64).view(self.out_channels, 1)
            base_w = (self.phase_w_base + k_idx.view(1, CK)) % T               # [Cout, CK]
        # self._gd.dump("phase_bases", base_x=base_x, base_w=base_w, k_idx=k_idx)

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

        # --- Step 11: fold back to feature map ---S
        y_cols = prod_bxbw 
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