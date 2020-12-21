import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SwitchNorm import SwitchNorm2d


def make_layers(
    cfg, block, in_chns=None, norm=True, skip=False, downsampling="maxpooling", upsampling="transconv"
):
    layers = []
    cur_chns = in_chns
    for ii, v in enumerate(cfg):
        if v == "M":
            if downsampling == "maxpooling":
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            elif downsampling == "strideconv":
                layers += [nn.Conv2d(cur_chns, cfg[ii + 1], kernel_size=3, stride=2, padding=1)]
                cur_chns = cfg[ii + 1]
            else:
                raise ValueError(f"unsupported down-sampling method")
        elif v == "U":
            if upsampling == "transconv":
                layers += [nn.ConvTranspose2d(cur_chns, cfg[ii + 1], kernel_size=4, stride=2, padding=1)]
                cur_chns = cfg[ii + 1]
            elif upsampling == "bilinear":
                layers += [nn.Upsample(scale_factor=2, mode="bilinear")]
            else:
                raise ValueError(f"unsupported up-sampling method")
        elif v == "D":
            layers += [nn.Dropout2d(p=0.3)]
        else:
            if ii == 0:
                layers += [DoubleConv(cur_chns, v, norm=norm)]
            else:
                in_v = 2 * v if skip else cur_chns
                layers += [block(in_v, v, norm=norm)]
            cur_chns = v
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, cfg, block, in_chns, norm=True, downsampling="maxpooling"):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.layers = make_layers(self.cfg, block, in_chns, norm=norm, downsampling=downsampling)

    def forward(self, xt, ht, cell_list, delta_t, split=False):
        kk = 0
        out = []
        if split:
            xt_out = []
        for v, module in zip(self.cfg, self.layers.children()):
            xt = module(xt)
            if isinstance(v, int):
                if split:
                    xt_out.append(xt)
                xt = cell_list[kk](xt, ht[kk], delta_t)
                kk += 1
                out.append(xt)
        if split:
            return out, xt_out
        else:
            return out


class Decoder(nn.Module):
    def __init__(self, cfg, in_chns, skip=True, norm=True, upsampling="transconv"):
        super(Decoder, self).__init__()
        self.cfg = ["U" if v == "M" else v for v in cfg[::-1]]
        self.cfg = self.cfg[1:]
        self.skip = skip
        self.layers = make_layers(self.cfg, SingleConv, in_chns, norm=norm, skip=skip, upsampling=upsampling)

    def forward(self, x):
        out = []
        kk = -1
        for ii, (v, module) in enumerate(zip(self.cfg, self.layers.children())):
            if ii == 0:
                y = module(x[kk])
                kk -= 1
            elif isinstance(v, int):
                if self.skip:
                    y = torch.cat((y, x[kk]), dim=1)
                    kk -= 1
                y = module(y)
                out.append(y)
            else:
                y = module(y)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_chns, out_chns, norm=True):
        super(DoubleConv, self).__init__()
        self.layers = nn.Sequential(
            SingleConv(in_chns, out_chns, norm=norm), SingleConv(out_chns, out_chns, norm=norm)
        )

    def forward(self, x):
        return self.layers(x)


class DoublePreActConv(nn.Module):
    def __init__(self, in_chns, out_chns, norm=True):
        super(DoublePreActConv, self).__init__()
        self.layers = nn.Sequential(
            SinglePreActConv(in_chns, out_chns, norm=norm), SinglePreActConv(out_chns, out_chns, norm=norm)
        )

    def forward(self, x):
        return self.layers(x)


class ResDoubleConv(nn.Module):
    def __init__(self, in_chns, out_chns, norm=True):
        super(ResDoubleConv, self).__init__()
        self.block = DoublePreActConv(in_chns, out_chns, norm=norm)

    def forward(self, x):
        y = self.block(x)
        delta_c = y.size(1) - x.size(1)
        x_skip = F.pad(x, (0, 0, 0, 0, 0, delta_c)) if delta_c > 0 else x
        return y + x_skip


class SingleConv(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size=3, stride=1, padding=1, norm=True, act=True):
        super(SingleConv, self).__init__()
        layers = [
            nn.Conv2d(
                in_chns, out_chns, kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm
            ),
        ]
        if norm:
            layers.append(SwitchNorm2d(out_chns))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SingleDeconv(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size=4, stride=2, padding=1, norm=True, act=True):
        super(SingleDeconv, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                in_chns, out_chns, kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm
            ),
        ]
        if norm:
            layers.append(SwitchNorm2d(out_chns))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SinglePreActConv(nn.Module):
    def __init__(self, in_chns, out_chns, norm=True):
        super(SinglePreActConv, self).__init__()
        if norm:
            layers = [SwitchNorm2d(in_chns), nn.ReLU(inplace=True)]
        else:
            layers = [nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(in_chns, out_chns, kernel_size=3, padding=1, bias=not norm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DSVBlock(nn.Module):
    def __init__(self, in_chns, out_chns, scale_factor):
        super(DSVBlock, self).__init__()
        layers = [
            nn.Conv2d(in_chns, out_chns, kernel_size=1, stride=1, padding=0),
        ]
        if scale_factor > 1:
            layers += [nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DSVLayer(nn.Module):
    def __init__(self, nb_filter, out_chns, num_layers=1):
        super(DSVLayer, self).__init__()
        self.num_layers = num_layers
        layers = []
        for ii in range(self.num_layers):
            layers.append(DSVBlock(nb_filter[ii], out_chns, scale_factor=2 ** ii))
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        out = []
        for ii in range(1, self.num_layers + 1):
            out.append(self.block[ii - 1](x[-ii]))
        y = out[0]
        for ii in range(1, self.num_layers):
            y += out[ii]
        return y


def init_weights(net, init_type="normal"):
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
