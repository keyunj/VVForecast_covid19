"""
This file (calc_feature_importance.py) is designed for:
    calculate feature importance by averaging
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import sys
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

sys.path.append("..")
from GLD.utils import AverageMeter


def calc_feature_importance(res_dir):
    feat_imps = AverageMeter()

    for phase in ["test"]:
        name_lst = np.loadtxt(osp.join(res_dir, phase, "0-results.txt"), dtype=str, skiprows=1, usecols=0)
        for name in name_lst:
            feat_imps.update(np.loadtxt(osp.join(res_dir, name.replace(".png", ".txt"))))

    np.savetxt(osp.join(res_dir, "feature_importance.txt"), feat_imps.avg, fmt="%.3f")

    return feat_imps


if __name__ == "__main__":
    prefix = "/home/dyj/disk1/covid/for-github/LesionDevelopment/output"
    res_dir = osp.join(prefix, "cald_23")

    feat_imps = calc_feature_importance(res_dir)
    feat_name = np.array(["Original intensity", "Lesion mask", "Lesion center distance map", "Lung margin distance map", "Vessel mask", "Vessel centerline distance map"])
    new_indices = np.array([3, 5, 4, 0, 2, 1])

    fig = plt.figure()
    y_pos = np.arange(len(feat_name))
    plt.barh(
        y_pos, feat_imps.avg[new_indices], color="#bbbbbb", align="center", alpha=0.4,
    )
    plt.xticks(np.arange(0, 0.3, 0.05))
    plt.yticks(y_pos, feat_name[new_indices])
    plt.xlabel("Feature Importance")
    plt.savefig(osp.join(res_dir, "feature_importance.png"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
