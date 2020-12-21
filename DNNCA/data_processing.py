import numpy as np
import torch
import torch.nn.functional as F

M_DAYS = {
    0: 0,
    1: 31,
    2: 59,
    3: 90,
    4: 120,
    5: 151,
    6: 181,
    7: 212,
    8: 243,
    9: 273,
    10: 304,
    11: 334,
}


def calc_interval(date_lst):
    month = [int(x.split("-")[0]) for x in date_lst]
    day = [int(x.split("-")[1]) for x in date_lst]
    day = [M_DAYS[m - 1] + d for m, d in zip(month, day)]
    day = np.array([d - day[0] for d in day])
    return np.abs(day)


def load_all_factors(name_lst, keys, center, ref="filterlesions", minus=False):
    out = []
    mask = []
    for k in keys:
        if k == "multilesions" or k == "filterlesions" or k == "deltalesions":
            data = np.stack([np.load(x.replace(ref, k)) for x in name_lst], axis=-1).astype(np.float32)
        else:
            data = np.stack([np.load(x.replace(ref, k)[:-7] + ".npy") for x in name_lst], axis=-1).astype(np.float32)
        if k == "lungsegs":
            mask.append(data)
            continue
        elif k == "lungdists" or k == "vesseldists" or k == "lesiondists":
            if minus:
                data = 1.0 - data / 255
            else:
                data /= 255
        # data = (data - data.min()) / (data.max() - data.min()).clip(0.001, None)
        if "lesions" not in k and k != "vessels":
            data = (data - data.mean()) / data.std().clip(0.001, None) + np.random.randn(*(data.shape)) * 0.1
        out.append(data)
    out = [mask[0] * x for x in out]
    out = np.concatenate(out, axis=-1)
    return out, mask[0][..., center].astype(np.uint8)
    # return out + np.random.randn(*(out.shape)) * 0.1, mask[0][..., center].astype(np.uint8)


def reshape_flatten(input_arr, kernel_size=1):
    ndim = input_arr.ndim - 1
    if len(kernel_size) == 1:
        kernel_size = (kernel_size,) * ndim
    elif len(kernel_size) != ndim:
        raise ValueError(f"the kernel size is not consist with input array")

    vol_shape = input_arr.shape[:-1]
    if np.all(kernel_size == 1):
        return np.squeeze(input_arr.reshape(np.prod(vol_shape), -1))
    else:
        padding = [x // 2 for x in kernel_size]
        to_tensor = torch.from_numpy(input_arr.transpose((-1,) + tuple(range(ndim))))[None].contiguous()
        tensor_view = F.unfold(to_tensor, kernel_size, padding=padding).transpose(1, 2)
        tensor_view = tensor_view.view(np.prod(vol_shape), -1)
        return tensor_view.numpy()


def random_select_samples(x_pool, y_pool, mask=None, sampler_p=0.5):
    numel = len(y_pool) if mask is None else mask.sum()
    pos_N = (y_pool > 0).sum()
    neg_N = numel - pos_N
    #
    # pos_indices = np.where(y_pool > 0)
    # if mask is None or (y_pool == 0 & mask).sum() < pos_N:
    #     neg_indices = np.where(y_pool == 0)
    # else:
    #     neg_indices = np.where(y_pool == 0 & mask)
    # indices = np.arange(len(neg_indices[0]))
    # indices = np.random.choice(indices, min(len(indices), 4 * pos_N))
    # neg_indices = tuple([x[indices] for x in neg_indices])
    # x_samples = np.concatenate((x_pool[pos_indices], x_pool[neg_indices]), axis=0)
    # y_samples = np.concatenate((y_pool[pos_indices], y_pool[neg_indices]), axis=0)
    # return x_samples, y_samples, np.ones_like(y_samples)

    indices = np.where(mask > 0)
    x_samples = x_pool[indices]
    y_samples = y_pool[indices]
    mask = mask[indices]
    neg_weight = sampler_p * pos_N / neg_N
    sample_weight = (mask != 0) * sampler_p + (mask == 0) * neg_weight
    return x_samples, y_samples, sample_weight

    # neg_weight = sampler_p * pos_N / neg_N
    # sample_weight = (y_pool != 0) * sampler_p + (y_pool == 0) * neg_weight
    # if mask is not None:
    #     sample_weight *= mask > 0.5
    # return x_pool.reshape(y_pool.size, -1), y_pool.reshape(-1), sample_weight.reshape(-1)
