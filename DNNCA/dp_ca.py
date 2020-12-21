"""
This file (dp_ca_23.py) is designed for:
    predict future lesion based on the suitable probability generated by deep learning method
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
import json
import argparse
import imageio
from itertools import repeat
from multiprocessing import Pool

from data_processing import calc_interval, reshape_flatten
from cell_automata import CellAutomata
from markov import Markov
from evaluation_ca import flat_to_single


def estimate_ca(driven_arr, mask, lesion, ma, delta_interval, half, interval):
    img_shape = lesion.shape
    ca = CellAutomata(
        num_classes=3,
        init_state=lesion,
        mask=mask.astype(np.bool),
        markov_T=ma,
        iter_times=max(1, delta_interval // half),
        interval=interval,
        target_t=delta_interval,
    )
    result = ca.evolve(driven_factors=driven_arr, iter_times=max(1, delta_interval // half))
    return result


def parallel_predict_single(les_lst, center, driven_dir, tar_dir, args):
    src_name = les_lst[2][center].replace("data", f"output/{driven_dir}")
    if not osp.exists(src_name):
        return

    save_name = les_lst[2][center].replace("data", f"output/{tar_dir}")

    if not osp.isdir(osp.dirname(save_name)):
        try:
            os.makedirs(osp.dirname(save_name))
        except:
            pass
    if osp.exists(save_name):
        return

    pre_mask = np.load(les_lst[0][center].replace("multilesions", "lungsegs")[:-7] + ".npy").astype(np.uint8)
    cur_mask = np.load(les_lst[1][center].replace("multilesions", "lungsegs")[:-7] + ".npy").astype(np.uint8)
    tar_mask = np.load(les_lst[2][center].replace("multilesions", "lungsegs")[:-7] + ".npy").astype(np.uint8)

    combine_res = imageio.imread(les_lst[2][center].replace("data", f"output/{driven_dir}"))
    h, w = combine_res.shape[:2]

    pre_lesion = combine_res[h // 2 :, : w // 5]
    cur_lesion = combine_res[h // 2 :, w // 5 : 2 * w // 5]
    driven_2nd = combine_res[: h // 2, 3 * w // 5 : 4 * w // 5] / 255.0
    driven_3rd = combine_res[: h // 2, 4 * w // 5 :] / 255.0

    pre_lesion = np.argmax(pre_lesion, axis=-1)
    cur_lesion = np.argmax(cur_lesion, axis=-1)
    driven_2nd = np.concatenate((1 - driven_2nd.sum(-1, keepdims=True), driven_2nd[..., 1:]), axis=-1)
    driven_3rd = np.concatenate((1 - driven_3rd.sum(-1, keepdims=True), driven_3rd[..., 1:]), axis=-1)

    interval = calc_interval([x[center].split("/")[-2] for x in les_lst])
    ma = Markov(np.stack((pre_lesion, cur_lesion)), classes=np.arange(3), mask=pre_mask, fill_empty_classes=True)

    # ca process
    pred_2nd = estimate_ca(
        driven_2nd, pre_mask, pre_lesion, ma, interval[1] - interval[0], args.half, interval[1] - interval[0]
    )
    pred_3rd = estimate_ca(
        driven_3rd, cur_mask, cur_lesion, ma, interval[2] - interval[1], args.half, interval[1] - interval[0]
    )

    # save
    pred_2nd = (pred_2nd[..., None] == np.arange(3)) * 1.0
    pred_3rd = (pred_3rd[..., None] == np.arange(3)) * 1.0
    pred_2nd[..., 0] = 0
    pred_3rd[..., 0] = 0
    combine_res[h // 2 :, 3 * w // 5 : 4 * w // 5] = np.rint(pred_2nd) * 255
    combine_res[h // 2 :, 4 * w // 5 : 5 * w // 5] = np.rint(pred_3rd) * 255
    imageio.imwrite(save_name, combine_res)
    print(f"simulated {les_lst[2][center]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="manual_same15")
    parser.add_argument("-r", "--radius", default=3)
    parser.add_argument("-c", "--ckp", default="20200701sigmoid_tgru")
    parser.add_argument("--half", default=2, type=int)
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/covid/for-github/LesionDevelopment"

    center = args.radius // 2

    for phase in ["test"]:
        with open(osp.join(prefix, "data", f"{args.data}_{args.radius}", f"{phase}_slices.json"), "r",) as fp:
            les_names = json.load(fp)["multilesions"]

        with Pool(processes=32) as tp:
            tp.starmap(
                parallel_predict_single,
                zip(
                    les_names,
                    repeat(center),
                    repeat(osp.join(f"{tar_dir}_df", phase)),
                    repeat(osp.join(f"{tar_dir}_ca", phase)),
                    repeat(args),
                ),
            )
