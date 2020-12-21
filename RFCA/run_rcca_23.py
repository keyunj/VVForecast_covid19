"""
This file (rf_ca_23.py) is designed for:
    RF + CA, generate prediction on 2 and 3 stages.
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import json
import argparse
import numpy as np
from glob import glob
import imageio
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import cohen_kappa_score
from itertools import repeat
from multiprocessing import Pool as ThreadPool
from cell_automata import CellAutomata
from markov import Markov
from data_processing import calc_interval, load_all_factors, reshape_flatten, random_select_samples
from util import flat_to_single


def select_patient_samples(cur_les_lst, center, kernel_size=(1, 1), sampler_p=0.5, minus=False):
    train_samples = []
    for les_lst in cur_les_lst:
        # load data
        pre_factors, pre_mask = load_all_factors(les_lst[0], keys, center, ref="multilesions", minus=minus)
        cur_lesions = np.load(les_lst[1][center]).astype(np.uint8)
        # flatten
        pre_factors = reshape_flatten(pre_factors, kernel_size=kernel_size)
        pre_mask = pre_mask.reshape(-1)
        cur_lesions = cur_lesions.reshape(-1)
        # select samples
        samples = random_select_samples(pre_factors, cur_lesions, mask=pre_mask, sampler_p=sampler_p)
        train_samples.append(samples)
    train_samples = list(zip(*train_samples))
    train_samples = [np.concatenate(x, axis=0) for x in train_samples]
    return train_samples


def estimate_ca(model, factors, mask, lesions, class_ids, ma, delta_interval, half, interval):
    img_shape = lesions.shape
    # prediction
    factors = reshape_flatten(factors, kernel_size=kernel_size)
    mask = mask.reshape(-1)
    pred = model.predict_proba(factors) * mask[..., None]
    driven_arr = np.zeros([*img_shape, 3])
    driven_arr[..., class_ids] = pred.reshape(*img_shape, -1)
    # ca
    ca = CellAutomata(
        num_classes=3,
        init_state=lesions,
        mask=mask.astype(np.bool),
        markov_T=ma,
        iter_times=max(1, delta_interval // half),
        interval=interval,
        target_t=delta_interval,
    )
    result = ca.evolve(driven_factors=driven_arr, iter_times=max(1, delta_interval // half))
    return driven_arr, result


def predict_each_slice(
    model,
    cur_les_lst,
    center,
    prefix,
    class_ids,
    half=2,
    kernel_size=(1, 1),
    ckp=None,
    phase="valid",
    radius=3,
    minus=False,
):
    assert ckp is not None

    for les_lst in cur_les_lst:
        # loading datas
        interval = calc_interval([x[center].split("/")[-2] for x in les_lst])
        pre_factors, pre_mask = load_all_factors(les_lst[0], keys, center, ref="multilesions", minus=minus)
        cur_factors, cur_mask = load_all_factors(les_lst[1], keys, center, ref="multilesions", minus=minus)
        pre_lesions = np.load(les_lst[0][center]).astype(np.uint8)
        cur_lesions = np.load(les_lst[1][center]).astype(np.uint8)
        tar_lesions = imageio.imread(les_lst[2][center])[..., 1:] / 255
        tar_lesions = flat_to_single(tar_lesions)

        # markov
        ma = Markov(np.stack((pre_lesions, cur_lesions)), classes=np.arange(3), mask=pre_mask, fill_empty_classes=True,)

        # estimation
        driven_2nd, pred_2nd = estimate_ca(
            model,
            pre_factors,
            pre_mask,
            pre_lesions,
            class_ids,
            ma,
            interval[1] - interval[0],
            half,
            interval[1] - interval[0],
        )
        driven_3st, pred_3st = estimate_ca(
            model,
            cur_factors,
            cur_mask,
            cur_lesions,
            class_ids,
            ma,
            interval[2] - interval[1],
            half,
            interval[1] - interval[0],
        )

        # save
        pre_images = np.load(les_lst[0][center].replace("multilesions", "images")[:-7] + ".npy")
        cur_images = np.load(les_lst[1][center].replace("multilesions", "images")[:-7] + ".npy")
        tar_images = np.load(les_lst[2][center].replace("multilesions", "images")[:-7] + ".npy")
        pre_images = (pre_images - pre_images.min()) / (pre_images.max() - pre_images.min() + 1e-6)
        cur_images = (cur_images - cur_images.min()) / (cur_images.max() - cur_images.min() + 1e-6)
        tar_images = (tar_images - tar_images.min()) / (tar_images.max() - tar_images.min() + 1e-6)

        driven_2nd[..., 0] = 0
        driven_3st[..., 0] = 0
        row1 = np.concatenate(
            (
                pre_images[..., None].repeat(3, -1),
                cur_images[..., None].repeat(3, -1),
                tar_images[..., None].repeat(3, -1),
                driven_2nd,
                driven_3st,
            ),
            axis=1,
        )
        pred_2nd = (pred_2nd[..., None] == np.arange(3)) * 1.0
        pred_3st = (pred_3st[..., None] == np.arange(3)) * 1.0
        row2 = np.concatenate(
            (
                (pre_lesions[..., None] == np.arange(3)) * 1.0,
                (cur_lesions[..., None] == np.arange(3)) * 1.0,
                (tar_lesions[..., None] == np.arange(3)) * 1.0,
                pred_2nd,
                pred_3st,
            ),
            axis=1,
        )
        row2[..., 0] = 0

        save_arr = np.rint(np.concatenate((row1, row2), axis=0) * 255).astype(np.uint8)
        save_name = les_lst[2][center].replace("data", f"output/{ckp}")
        if not os.path.isdir(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))
        imageio.imwrite(save_name, save_arr)
        # save importance
        save_name = les_lst[2][center].replace("data", f"output/{ckp}").replace(".png", ".txt")
        feat_importance = np.array(model.feature_importances_).reshape(radius, -1, *kernel_size)
        feat_importance = feat_importance.sum(axis=(0, -1, -2))
        np.savetxt(save_name, feat_importance, fmt="%.5f")
        print(f"simulated {les_lst[2][center]}")


def parallel_predict_single(les_lst, center, prefix, args, kernel_size=(1, 1), sampler_p=0.5):
    src_name = les_lst[2][center]
    if not osp.exists(src_name):
        return

    # model = SVC(C=0.9)
    model = RandomForestClassifier(n_estimators=100, oob_score=True, class_weight="balanced_subsample")
    train_samples = select_patient_samples([les_lst], center, kernel_size=kernel_size, sampler_p=sampler_p)
    model = model.fit(train_samples[0], train_samples[1], sample_weight=train_samples[2])
    predict_each_slice(
        model,
        [les_lst],
        center,
        prefix,
        class_ids=np.unique(train_samples[1]),
        half=args.half,
        kernel_size=kernel_size,
        ckp=args.ckp,
        phase=args.phase,
        radius=args.radius,
        minus=args.minus,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="manual_same15")
    parser.add_argument("-r", "--radius", default=3)
    parser.add_argument("-p", "--phase", default="valid", help="phase")
    parser.add_argument("-c", "--ckp", default="rfca_23")
    parser.add_argument("--minus", dest="minus", action="store_true", default=False)
    parser.add_argument("--half", default=2, type=int)
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/covid/for-github/LesionDevelopment"

    keys = ["images", "multilesions", "lesiondists", "lungsegs", "lungdists", "vessels", "vesseldists"]

    with open(f"{prefix}/data/{args.data}_{args.radius}/{args.phase}_slices.json", "r") as fp:
        les_names = json.load(fp)["multilesions"]

    args.ckp = f"{args.ckp}/{args.phase}"

    center = args.radius // 2
    kernel_size = (15, 15)

    # for les_lst in les_names[:16]:
    #     parallel_predict_single(les_lst, center, prefix, args, kernel_size)

    # train on single slice
    with ThreadPool(processes=16) as tp:
        tp.starmap(
            parallel_predict_single, zip(les_names, repeat(center), repeat(prefix), repeat(args), repeat(kernel_size),),
        )

