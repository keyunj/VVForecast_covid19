import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .modules import *


__all__ = [
    "ForecastConvGRU",
    "TrueConvGRU",
    "generate_lesion",
]


class ForecastConvGRU(nn.Module):
    def __init__(self, args):
        super(ForecastConvGRU, self).__init__()
        self.radius = args.radius
        self.in_chns = args.in_chns
        self.out_chns = args.out_chns
        self.num_classes = args.num_classes
        self.in_size = args.in_size
        self.nb_filter = args.nb_filter
        self.num_layers = len(args.nb_filter)
        self.return_all = args.return_all
        self.USE_CUDA = args.use_cuda
        self.t_gate = args.t_gate

        kernel_size = self._extend_for_multilayer(args.kernel_size, self.num_layers)
        assert len(kernel_size) == self.num_layers

        cfg = [self.nb_filter[0]]
        for nb in self.nb_filter[1:]:
            if args.use_drop:
                cfg += ["M", "D", nb]
            else:
                cfg += ["M", nb]

        # encoder and decoder parts
        self.enc = FeatureExtract(cfg, self.in_chns)
        self.dec = FeatureRestore(cfg, self.nb_filter[-1], skip=args.skip)
        self.classifier = nn.Conv2d(self.nb_filter[0], self.out_chns * self.num_classes, kernel_size=1)

        # lstm parts
        cell_list = []
        cur_size = self.in_size
        for ii in range(self.num_layers):
            cell_list.append(
                ConvGRUCell(
                    cur_size,
                    input_dim=self.nb_filter[ii],
                    hidden_dim=self.nb_filter[ii],
                    kernel_size=kernel_size[ii],
                    bias=args.bias,
                    USE_CUDA=self.USE_CUDA,
                )
            )
            cur_size = [x // 2 for x in cur_size]
        self.cell_list = nn.ModuleList(cell_list)

        if self.t_gate:
            self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.beta.data[0] = 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.GroupNorm)
            ):
                init_weights(m, init_type="kaiming")
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2.0 / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()

    def forward(self, sample):
        hidden_state = self._init_hidden(batch_size=sample["data"].size(0))

        layer_out_lst = []

        seq_len = sample["data"].size(1)

        ht = hidden_state
        for t in range(seq_len):
            # encoder
            xt = self.enc(sample["data"][:, t])

            if self.t_gate and t > 0:
                w = torch.exp(
                    -torch.abs(self.beta) * (sample["interval"][:, t] - sample["interval"][:, t - 1] - 1)
                )

            # lstm, information propagation
            for layer_idx in range(self.num_layers):
                if self.t_gate and t > 0:
                    ht_w = ht[layer_idx] * w.view(-1, 1, 1, 1)
                else:
                    ht_w = ht[layer_idx]

                ht[layer_idx] = self.cell_list[layer_idx](input_tensor=xt[layer_idx], h_cur=ht_w)

            # decoder
            if self.return_all or t == seq_len - 1:
                pt = self.dec(ht)
                yt = self.classifier(pt)
                # store inner results
                layer_out_lst.append(yt)
        return layer_out_lst

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class TrueConvGRU(nn.Module):
    def __init__(self, args):
        super(TrueConvGRU, self).__init__()
        self.radius = args.radius
        self.in_chns = args.in_chns
        self.out_chns = args.out_chns
        self.num_classes = args.num_classes
        self.in_size = args.in_size
        self.nb_filter = args.nb_filter
        self.num_layers = len(args.nb_filter)
        self.return_all = args.return_all
        self.USE_CUDA = args.use_cuda
        self.t_gate = args.t_gate

        kernel_size = self._extend_for_multilayer(args.kernel_size, self.num_layers)
        assert len(kernel_size) == self.num_layers

        cfg = [self.nb_filter[0]]
        for nb in self.nb_filter[1:]:
            cfg += ["M", nb]

        # encoder and decoder parts
        self.enc = Encoder(cfg, ResDoubleConv, self.in_chns, downsampling="strideconv")
        self.dec = Decoder(cfg, cfg[-1], skip=args.skip, upsampling="transconv")
        self.classifier = DSVLayer(
            self.nb_filter, self.out_chns * self.num_classes, num_layers=args.deep_layers
        )

        # lstm parts
        cell_list = []
        cur_size = self.in_size
        for ii in range(self.num_layers):
            cell_list.append(
                ConvGRUCell(
                    cur_size,
                    input_dim=self.nb_filter[ii],
                    hidden_dim=self.nb_filter[ii],
                    t_gate=self.t_gate,
                    kernel_size=kernel_size[ii],
                    bias=args.bias,
                    USE_CUDA=self.USE_CUDA,
                )
            )
            cur_size = [x // 2 for x in cur_size]
        self.cell_list = nn.ModuleList(cell_list)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.GroupNorm)
            ):
                init_weights(m, init_type="kaiming")

    def forward(self, sample):
        hidden_state = self._init_hidden(batch_size=sample["data"].size(0))

        layer_out_lst = []

        seq_len = sample["data"].size(1)

        ht = hidden_state
        for t in range(seq_len):
            # encoder
            delta_t = (
                sample["interval"][:, t] - sample["interval"][:, t - 1] if self.t_gate and t > 0 else None
            )
            ht = self.enc(sample["data"][:, t], ht, self.cell_list, delta_t)
            # decoder
            if self.return_all or t == seq_len - 1:
                dec = self.dec(ht)
                y = self.classifier(dec)
                # store inner results
                layer_out_lst.append(y)
        return layer_out_lst

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, t_gate, kernel_size, bias, USE_CUDA):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size // 2
        self.t_gate = t_gate
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.USE_CUDA = USE_CUDA

        if self.t_gate:
            self.t_conv = nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=kernel_size,
                padding=self.padding,
                bias=self.bias,
            )

        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,  # for update_gate, reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_can = nn.Conv2d(
            in_channels=input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros([batch_size, self.hidden_dim, self.height, self.width])
        if self.USE_CUDA:
            hidden_state = hidden_state.cuda()
        return hidden_state

    def forward(self, input_tensor, h_cur, delta_t=None):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        if self.t_gate and delta_t is not None:
            t_cconv = self.t_conv(h_cur)
            t_conv = torch.tanh(t_cconv)
            c_info = h_cur - t_conv
            t_info = t_conv / torch.log(torch.exp(torch.tensor(1.0)) + delta_t.view(-1, 1, 1, 1))
            t_c_com = c_info + t_info
        else:
            c_info = h_cur
            t_c_com = h_cur

        combined = torch.cat([input_tensor, c_info], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)

        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * c_info], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * t_c_com + update_gate * cnm
        return h_next


def generate_lesion(preds, interval, num_classes=1, coef=None):
    # b, c, h, w
    pi = torch.tensor(np.pi)
    output = []
    for t in range(1, len(preds) + 1):
        b, c, h, w = preds[-t].size()
        cur_pred = preds[-t].view(b, num_classes, -1, h, w)
        x = (interval - interval[:, -t - 1 : -t]).view(b, 1, -1, 1, 1)

        mu = cur_pred[:, :, 0:1]
        logvar = cur_pred[:, :, 1:2]
        if coef is None:
            phi = 1
        elif coef == "tanh":
            phi = (torch.tanh(torch.exp(-logvar) * x + cur_pred[:, :, 2:3]) + 1) / 2
        elif coef == "softsign":
            phi = (F.softsign(torch.exp(-logvar) * x + cur_pred[:, :, 2:3]) + 1) / 2
        elif coef == "sigmoid":
            phi = torch.sigmoid(torch.exp(-logvar) * x + cur_pred[:, :, 2:3])
        else:
            raise ValueError(f"Unsupported coeffcient type {coef}")
        p = torch.exp(-torch.exp(-2 * logvar) * (x - mu) ** 2) * phi
        p = torch.cat((1 - p.sum((1), keepdim=True), p), dim=1).clamp_min(0.001)
        output.append(p)
    # b, k, t, h, w, from back to front
    return output
