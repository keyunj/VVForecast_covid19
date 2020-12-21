import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__ALL__ = [
    "CellAutomata",
    "markov_stop_criterion",
]


class CellAutomata(object):
    def __init__(
        self,
        num_classes,
        init_state=None,
        state_size=None,
        neig_mode="moore",
        neig_size=(3, 3),
        mask=None,
        markov_T=None,
        iter_times=None,
        interval=None,
        target_t=0,
    ):
        assert init_state is not None or state_size is not None
        if state_size is None:
            self.state_size = init_state.shape

        self.num_classes = num_classes
        self.neig_mode = neig_mode
        self.neig_size = neig_size
        self.center = [x // 2 for x in self.neig_size]
        self.mask = (
            torch.from_numpy(mask).view(1, 1, *self.state_size).contiguous() if mask is not None else None
        )
        self.markov_T = markov_T
        self.iter_times = max(iter_times, 3)
        self.interval = interval
        self.target_t = target_t

        self.initial_state(init=init_state)

        local_weight = np.ones(self.neig_size)
        for ii in range(1, self.center[0]):
            local_weight[ii:-ii, ii:-ii] += 1
        local_weight /= local_weight.max()
        if self.neig_mode == "moore":
            local_neig = np.ones(self.neig_size)
            local_neig[self.center[0], self.center[1]] = 0
        elif self.neig_mode == "von":
            local_neig = np.zeros(self.neig_size)
            local_neig[self.center, :] = 1
            local_neig[:, self.center] = 1
            local_neig[self.center[0], self.center[1]] = 0
        else:
            raise ValueError(f"not supported neighborhood mode {self.neig_mode}")
        self.neig_indices = np.ravel_multi_index(np.where(local_neig), self.neig_size).tolist()
        self.local_weight = local_weight.reshape(-1)[self.neig_indices]

        if self.markov_T is None and self.iter_times is None:
            self.iter_times = interval

    @property
    def target_area(self):
        transition_V = torch.from_numpy(self.markov_T.transitions / self.interval)
        target = (transition_V.sum((0)) - transition_V.sum((1))) * self.target_t + self.init_area
        return target.clamp_(0, self.init_area.sum())

    @property
    def current_area(self):
        indices = self.cell_state.view(-1)
        if self.mask is not None:
            return torch.zeros(self.num_classes).scatter_add_(0, indices.long(), self.mask.view(-1).float())
        else:
            return torch.zeros(self.num_classes).scatter_add_(
                0, indices.long(), torch.ones_like(indices).float()
            )

    def step_one(self):
        self.iter_cnt += 1

    @property
    def stop_criterion(self):
        if self.markov_T is not None:
            return self.markov_stop_criterion
        else:
            return self.multistep_stop_criterion

    @property
    def markov_stop_criterion(self):
        area_delta = torch.norm(self.current_area - self.target_area)
        # print(f"difference between current and target state is {area_delta}")
        if area_delta < 10 or self.iter_cnt > self.iter_times:
            return True
        return False

    @property
    def multistep_stop_criterion(self):
        if self.iter_cnt > self.iter_times:
            return True
        return False

    def initial_state(self, init=None):
        # only batch-image are supported
        assert np.all(init.shape == self.state_size)
        if init is not None:
            self.cell_state = torch.from_numpy(init).view(1, 1, *self.state_size).contiguous().float()
        else:
            self.cell_state = (
                torch.randint(0, self.num_classes, (1, 1, *self.state_size)).contiguous().float()
            )
        self.init_area = self.current_area

    def evolve(self, driven_factors, iter_times=None):
        assert (
            np.all(driven_factors.shape[:-1] == self.state_size)
            and driven_factors.shape[-1] == self.num_classes
        )
        self.iter_cnt = 0
        iter_times = self.iter_times if iter_times is None else iter_times

        driven_factors_view = torch.from_numpy(driven_factors).float().view(-1, self.num_classes)

        # while not self.stop_criterion:
        while self.iter_cnt < iter_times:
            cell_state_view = F.unfold(self.cell_state, self.neig_size, padding=self.center).transpose(1, 2)
            cell_state_view = cell_state_view.view(-1, np.prod(self.neig_size))
            # random factor
            factor_eps = torch.rand((cell_state_view.size(0), self.num_classes))
            # factor_eps = 1 / (1 + torch.exp(-3 * torch.rand((cell_state_view.size(0), self.num_classes))))
            factor_eps = factor_eps / factor_eps.sum(dim=1, keepdim=True).clamp_min(0.001)
            # suitable factor
            factor_suit = driven_factors_view
            # adjacent influence factor
            factor_neig = (
                torch.ones_like(driven_factors_view).scatter_add_(
                    1,
                    cell_state_view[:, self.neig_indices].long(),
                    torch.from_numpy(self.local_weight[None])
                    .expand_as(cell_state_view[:, self.neig_indices])
                    .float(),
                )
                / self.local_weight.sum()
            )
            # factor_neig = 1 / (1 + torch.exp(-factor_neig))
            # multiple all factors
            transition = (
                (factor_eps * factor_neig * factor_suit)
                .transpose(0, 1)
                .view(1, self.num_classes, *self.state_size)
            )
            mask = transition.max((1), keepdim=True)[0] > 0.05  # 04222kernel 0.05, 04223kernel 0.0
            # mask = (
            #     factor_suit.transpose(0, 1)
            #     .view(1, self.num_classes, *self.state_size)[:, 1:]
            #     .max((1), keepdim=True)[0]
            #     >= 0.4
            # )
            if self.mask is not None:
                mask = mask & self.mask
            # transist to the new state
            self.cell_state[mask] = torch.argmax(transition, dim=1, keepdim=True)[mask].float()
            # step once
            self.step_one()

        return self.cell_state.char().squeeze().numpy()
