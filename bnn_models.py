# bnn_models.py
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.nn.quantized as nnq
import torch.ao.quantization.stubs as stubs
import torch.ao.nn.intrinsic.quantized as nniq
import torch.nn.functional as F
import argparse, copy, random, numpy as np
from torch.ao.quantization import QuantStub, DeQuantStub
import torch.ao.nn.qat as nnqat

# -----------------
# 模型本體
# -----------------
def make_vgg16_features(binarize_from=1):
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

# def make_vgg16_features(binarize_from=1):
#     cfg=[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
#     layers=[]; in_ch=3; ci=0
#     for v in cfg:
#         if v=='M': layers.append(nn.MaxPool2d(2,2)); continue
#         if ci>=binarize_from: layers += [ BinaryConv2d(in_ch,v,3,1,1,False),nn.BatchNorm2d(v),BinaryAct()]
#         else:                 layers += [nn.Conv2d(in_ch,v,3,1,1,False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#         in_ch=v; ci+=1
#     return nn.Sequential(*layers)

# def make_vgg16_features(binarize_from=1):
#     cfg=[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']
#     layers=[]; in_ch=3; ci=0
#     for v in cfg:
#         if v=='M': layers.append(nn.MaxPool2d(2,2)); continue
#         if ci>=binarize_from: layers += [ BinaryConv2d(in_ch,v,3,1,1,False),nn.BatchNorm2d(v),BinaryAct()]
#         else:                 layers += [nn.Conv2d(in_ch,v,3,1,1,False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#         in_ch=v; ci+=1
#     return nn.Sequential(*layers)

class BinaryVGG16(nn.Module):
    def __init__(self,num_classes=10,binarize_from=1,dropout=0.0):
        super().__init__()
        self.features=make_vgg16_features(binarize_from)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(512,num_classes)) if dropout>0 else nn.Linear(512,num_classes)
    def forward(self,x):
        x=self.features(x); x=self.pool(x); x=torch.flatten(x,1); return self.classifier(x)


class BNNQuantWrapper(nn.Module):
    """只量化 head/cls，保留中段二值區塊不量化。

    head  = 所有「實數層」，到第一個 binary block 之前為止
    tail  = 第一個 binary block 開始到結尾

    第一個 binary block 定義：
      - 若 pattern 是 BinaryAct -> BinaryConv2d -> ...，
        則從這顆 BinaryAct 開始算 binary block
      - 否則就從第一顆 BinaryConv2d 開始
    """
    def __init__(self, bnn: BinaryVGG16):
        super().__init__()

        # 先把 features 展成 list，方便用 index 看前後關係
        feats = [m for m in bnn.features]
        first_bin = len(feats)  # 預設：沒有 binary block

        for i, m in enumerate(feats):
            if isinstance(m, BinaryConv2d):
                # 如果前一層就是 BinaryAct，就把「那顆 BinaryAct」也算進 binary block
                if i > 0 and isinstance(feats[i - 1], BinaryAct):
                    first_bin = i - 1
                else:
                    first_bin = i
                break

        # head: [0, first_bin)   tail: [first_bin, end)
        self.head = nn.Sequential(*feats[:first_bin])
        self.tail = nn.Sequential(*feats[first_bin:])

        self.pool = bnn.pool
        self.cls = bnn.classifier

        # 量化 stub（跟原本一樣）
        self.q_in, self.dq_head = tq.QuantStub(), tq.DeQuantStub()
        self.q_cls, self.dq_out = tq.QuantStub(), tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_in(x)
        x = self.head(x)
        x = self.dq_head(x)

        x = self.tail(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.q_cls(x)
        x = self.cls(x)
        x = self.dq_out(x)
        return x


class BinaryConv2d(nn.Module):
    """不繼承 nn.Conv2d；用 F.conv2d 並**明確**傳參，避免底層型別不匹配。"""
    def __init__(self,in_ch,out_ch,k=3,stride=1,padding=1,bias=False,groups=1,dilation=1,use_alpha=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_ch, in_ch//groups, k, k))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = (stride, stride) if isinstance(stride,int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding,int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation,int) else tuple(dilation)
        self.groups = int(groups)
        self.use_alpha=use_alpha
    def forward(self,x):
        w=self.weight
        if self.use_alpha:
            alpha=w.detach().abs().mean(dim=(1,2,3),keepdim=True)
            wb=sign(w); wq=(wb*alpha).clamp(-1,1)
        else:
            wq=sign(w)
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

class BinaryAct(nn.Module):
    def forward(self,x):
        y=sign(x)
        return torch.where(y==0, torch.ones_like(y), y)

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x); return x.sign()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).float()

def sign(x): return SignSTE.apply(x)
# -----------------
# 量化輔助
# -----------------
def choose_backend(name):
    name=(name or 'fbgemm').lower()
    if name not in ('fbgemm','qnnpack'): name='fbgemm'
    torch.backends.quantized.engine=name
    return name

def make_qconfig(mode, backend, per_channel):
    if mode=='qat':
        base=tq.get_default_qat_qconfig(backend)
        if per_channel:
            w=tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
            return tq.QConfig(activation=base.activation, weight=w)
        return base
    act=tq.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    w=tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric) if per_channel \
        else tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    return tq.QConfig(activation=act, weight=w)

def fuse_head(seq: nn.Sequential):
    # 反覆掃描，直到沒有可 fuse 的為止
    changed = True
    while changed:
        changed = False
        mods = list(seq.named_children())
        i = 0
        while i < len(mods):
            name, m = mods[i]
            # Conv + BN + ReLU
            if isinstance(m, nn.Conv2d):
                if i + 2 < len(mods) and isinstance(mods[i+1][1], nn.BatchNorm2d) and isinstance(mods[i+2][1], nn.ReLU):
                    tq.fuse_modules(seq, [mods[i][0], mods[i+1][0], mods[i+2][0]], inplace=True)
                    changed = True
                    break  # 重新掃描
                # Conv + BN
                if i + 1 < len(mods) and isinstance(mods[i+1][1], nn.BatchNorm2d):
                    tq.fuse_modules(seq, [mods[i][0], mods[i+1][0]], inplace=True)
                    changed = True
                    break
                # Conv + ReLU
                if i + 1 < len(mods) and isinstance(mods[i+1][1], nn.ReLU):
                    tq.fuse_modules(seq, [mods[i][0], mods[i+1][0]], inplace=True)
                    changed = True
                    break
            i += 1

def _set_qconfig_tree(model, qcfg):
    for m in model.modules(): m.qconfig=None
    model.qconfig=qcfg
    # 防呆：中段二值層永遠不量化
    for m in model.modules():
        if isinstance(m, (BinaryConv2d, BinaryAct)):
            m.qconfig=None

def _force_qat_modules(wrapped: BNNQuantWrapper, qcfg):
    # head: Conv2d → QAT.Conv2d
    for name, mod in list(wrapped.head.named_children()):
        if isinstance(mod, nn.Conv2d):
            mod.qconfig = qcfg
            qat = nnqat.Conv2d.from_float(mod)
            setattr(wrapped.head, name, qat)
    # cls: Linear → QAT.Linear
    if isinstance(wrapped.cls, nn.Linear):
        wrapped.cls.qconfig = qcfg
        wrapped.cls = nnqat.Linear.from_float(wrapped.cls)
    elif isinstance(wrapped.cls, nn.Sequential):
        for n,mod in list(wrapped.cls.named_children()):
            if isinstance(mod, nn.Linear):
                mod.qconfig = qcfg
                setattr(wrapped.cls, n, nnqat.Linear.from_float(mod))
                break

def convert_to_int8(prepared: nn.Module) -> nn.Module:
    try:
        import torch.fx as fx
        if isinstance(prepared, fx.GraphModule):
            return convert_fx(prepared.eval().cpu())
    except Exception:
        pass
    return tq.convert(prepared.eval().cpu(), inplace=False)

def prepare_int8_model(float_model: nn.Module, args):
    backend = choose_backend(getattr(args, 'quant_backend', 'fbgemm'))
    qcfg    = make_qconfig(getattr(args, 'int8', 'qat'),
                           backend, getattr(args, 'per_channel', False))

    if getattr(args, 'binary', True):
        # ---- 以 BNN wrapper 只量化：輸入端(head)與最後 classifier ----
        wrapped = BNNQuantWrapper(float_model)

        # 1) 先 eval 再 fuse（BN 只能在 eval 被折疊）
        wrapped.eval()
        fuse_head(wrapped.head)
        # 保險檢查：head 內不應再有 BN
        assert not any(isinstance(m, nn.BatchNorm2d) for m in wrapped.head.modules()), \
            "fuse_head 失敗：head 中仍含 BatchNorm2d"

        # 2) 設定 qconfig
        #    先清空/指定全域 qconfig，並排除 Binary 區段
        _set_qconfig_tree(wrapped, qcfg)

        #    明確把 qconfig 指到 head/cls 的可量化子層與兩個 Stub
        for m in wrapped.head.modules():
            if isinstance(m, (nn.Conv2d, nn.ReLU, nn.ReLU6, nn.Sequential)):
                m.qconfig = qcfg
        for m in wrapped.cls.modules():
            if isinstance(m, (nn.Linear, nn.ReLU, nn.ReLU6, nn.Sequential)):
                m.qconfig = qcfg
        wrapped.q_in.qconfig  = qcfg
        wrapped.q_cls.qconfig = qcfg
        tq.propagate_qconfig_(wrapped)

        # 3) 將需要量化的層換成 QAT 版（Conv/Linear）
        _force_qat_modules(wrapped, qcfg)

        # 4) 準備量化：QAT 需在 train 狀態；PTQ 保持 eval
        if getattr(args, 'int8', 'qat') == 'qat':
            wrapped.train()                     # prepare_qat 僅在 training 模式可用
            tq.prepare_qat(wrapped, inplace=True)
        else:
            tq.prepare(wrapped, inplace=True)
            wrapped.eval()

        return wrapped
    else:
        # 非 BNN：示範 FX 版
        ex=torch.randn(1,3,32,32)
        qmap=QConfigMapping().set_global(qcfg)
        return prepare_qat_fx(float_model, qmap, ex) if getattr(args,'int8','qat')=='qat' else prepare_fx(float_model, qmap, ex)

def _as_2tuple(v):
    if isinstance(v, tuple): return (int(v[0]), int(v[1]))
    v=int(v); return (v,v)

# 你原先用到的保險函式（維持介面）
def _sanitize_conv_hparams(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            s=_as_2tuple(m.stride);   s=(max(1,s[0]),max(1,s[1]))
            p=_as_2tuple(m.padding);  p=(max(0,p[0]),max(0,p[1]))
            d=_as_2tuple(m.dilation); d=(max(1,d[0]),max(1,d[1]))
            g=int(getattr(m,'groups',1)); 
            if g<=0 or m.in_channels%g!=0: g=1
            m.stride, m.padding, m.dilation, m.groups = s,p,d,g