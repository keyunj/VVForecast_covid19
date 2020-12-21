"""
This file (calc_subgroup_result.py) is designed for:
    calculate results of sub-group defined by age, sex and time interval
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
import argparse


def print_save(mask, metrics, title, fname):
    dice_all, dice_l1, dice_l2, kappa, FoM, ldr_all, ldr_l1, ldr_l2 = metrics

    print_str = [title]
    print_str.append(f"the average dice for the whole is {dice_all[0].mean()} {dice_all[1].mean()}")
    print_str.append(f"the average dice for lesion 1 is {dice_l1[0].mean()} {dice_l1[1].mean()}")
    print_str.append(f"the average dice for lesion 2 is {dice_l2[0].mean()} {dice_l2[1].mean()}")
    print_str.append(f"the average kappa for the whole is {kappa[0].mean()} {kappa[1].mean()}")
    print_str.append(f"the average FoM for the whole is {FoM[0].mean()} {FoM[1].mean()}")
    print_str.append(f"the average LDR for the whole is {ldr_all[0].mean()} {ldr_all[1].mean()}")
    print_str.append(f"the average LDR for lesion 1 is {ldr_l1[0].mean()} {ldr_l1[1].mean()}")
    print_str.append(f"the average LDR for lesion 2 is {ldr_l2[0].mean()} {ldr_l2[1].mean()}")

    print("\n".join(print_str))
    with open(fname, "a") as fp:
        fp.write("\n".join(print_str) + "\n")


def calc_subgroup_result(res_dir, phase="valid"):
    all_results = np.loadtxt(osp.join(res_dir, phase, "0-results.txt"), skiprows=1, usecols=range(1, 20))
    dice_all, dice_l1, dice_l2, kappa, FoM, ldr_all, ldr_l1, ldr_l2 = {}, {}, {}, {}, {}, {}, {}, {}
    # meta
    age, sex, interval = all_results[:, 0], all_results[:, 1], all_results[:, 2]
    # first stage
    dice_all[0], dice_l1[0], dice_l2[0] = all_results[:, 3], all_results[:, 4], all_results[:, 5]
    kappa[0], FoM[0] = all_results[:, 6], all_results[:, 7]
    ldr_all[0], ldr_l1[0], ldr_l2[0] = all_results[:, 8], all_results[:, 9], all_results[:, 10]
    # second stage
    dice_all[1], dice_l1[1], dice_l2[1] = all_results[:, 11], all_results[:, 12], all_results[:, 13]
    kappa[1], FoM[1] = all_results[:, 14], all_results[:, 15]
    ldr_all[1], ldr_l1[1], ldr_l2[1] = all_results[:, 16], all_results[:, 17], all_results[:, 18]

    metrics = (dice_all, dice_l1, dice_l2, kappa, FoM, ldr_all, ldr_l1, ldr_l2)

    # plain
    mask = age <= 50
    print_save(mask, metrics, "Age less equal 50", osp.join(res_dir, phase, "metrics.txt"))
    # mask = (age > 40) & (age <= 60)
    # print_save(mask, metrics, "Age lies in (40,60]", osp.join(res_dir, phase, "metrics.txt"))
    mask = age > 50
    print_save(mask, metrics, "Age greater 50", osp.join(res_dir, phase, "metrics.txt"))
    mask = sex == 0
    print_save(mask, metrics, "Male", osp.join(res_dir, phase, "metrics.txt"))
    mask = sex == 1
    print_save(mask, metrics, "Female", osp.join(res_dir, phase, "metrics.txt"))
    mask = interval <= 7
    print_save(mask, metrics, "Interval less than 7 days", osp.join(res_dir, phase, "metrics.txt"))
    mask = interval > 7
    print_save(mask, metrics, "Interval geater 7 days", osp.join(res_dir, phase, "metrics.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckp", default="cald_23")
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/covid/for-github/LesionDevelopment/output"
    res_dir = osp.join(prefix, args.ckp)

    for phase in ["test"]:
        calc_subgroup_result(res_dir, phase)
