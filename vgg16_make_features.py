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