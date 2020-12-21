import os
import os.path as osp
import json
import pydicom
import imageio
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import sys

sys.path.append("..")
from GLD.utils import AverageMeter, cal_dice, Logger
from data_processing import calc_interval

SEX = {"M": 0, "F": 1}


def get_age_sex(path):
    dcm_name = glob(f"{path}/I*")[1]
    ds = pydicom.dcmread(dcm_name, force=True)
    age = int(ds.StudyDate[:4]) - int(ds.PatientBirthDate[:4])
    sex = SEX[ds.PatientSex]
    return age, sex


def area_change_ratio(pred_arr, cur_arr, tar_arr):
    all_delta_c = ((pred_arr > 0).sum() - (tar_arr > 0).sum()) / (tar_arr > 0).sum().clip(1, None)
    les1_delta_c = ((pred_arr == 1).sum() - (tar_arr == 1).sum()) / (tar_arr == 1).sum().clip(1, None)
    les2_delta_c = ((pred_arr == 2).sum() - (tar_arr == 2).sum()) / (tar_arr == 2).sum().clip(1, None)
    # abs
    all_delta_c = np.abs(all_delta_c).clip(0, 1)
    les1_delta_c = np.abs(les1_delta_c).clip(0, 1)
    les2_delta_c = np.abs(les2_delta_c).clip(0, 1)
    return all_delta_c, les1_delta_c, les2_delta_c


def location_change_ratio(pred_arr, tar_arr):
    sub_indices = [np.arange(x) for x in pred_arr.shape]
    indices = np.meshgrid(*sub_indices, indexing="ij")
    # gt
    all_gt_mass = [((tar_arr > 0) * x).mean() for x in indices]
    l1_gt_mass = [((tar_arr == 1) * x).mean() for x in indices]
    l2_gt_mass = [((tar_arr == 2) * x).mean() for x in indices]
    # pred
    all_p_mass = [((pred_arr > 0) * x).mean() for x in indices]
    l1_p_mass = [((pred_arr == 1) * x).mean() for x in indices]
    l2_p_mass = [((pred_arr == 2) * x).mean() for x in indices]
    # distance
    all_d = np.linalg.norm(np.array(all_p_mass) - np.array(all_gt_mass))
    l1_d = np.linalg.norm(np.array(l1_p_mass) - np.array(l1_gt_mass))
    l2_d = np.linalg.norm(np.array(l2_p_mass) - np.array(l2_gt_mass))
    return all_d, l1_d, l2_d


def coef_FoM(pred_arr, cur_arr, tar_arr):
    gt_c = (cur_arr != tar_arr) * 1.0
    x1 = (pred_arr != cur_arr) * gt_c
    x2 = (pred_arr != cur_arr) * (1 - gt_c)
    x3 = (pred_arr == cur_arr) * gt_c
    x4 = (pred_arr == cur_arr) * (1 - gt_c)
    false_x1 = (pred_arr != tar_arr) * x1
    A = x3.sum()
    B = x1.sum()
    C = false_x1.sum()
    D = x2.sum()
    return B / (A + B + C + D).clip(1, None)


def flat_to_single(input_arr):
    arr = np.concatenate((1 - input_arr.sum((-1), keepdims=True), input_arr), axis=-1)
    return np.argmax(arr, axis=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="manual_same15")
    parser.add_argument("-r", "--radius", default=3)
    parser.add_argument("-p", "--phase", default="valid")
    parser.add_argument("-c", "--ckp", default="cald_23")
    args = parser.parse_args()

    args.ckp = f"{args.ckp}/{args.phase}"

    prefix = "/home/dyj/disk1/covid/for-github/LesionDevelopment"
    res_dir = osp.join(prefix, "output", args.ckp)

    with open(f"{prefix}/data/{args.data}_{args.radius}/{args.phase}_slices.json", "r") as fp:
        les_names = json.load(fp)["multilesions"]
    les_names = [[y[1] for y in x] for x in les_names]

    logger = Logger(osp.join(res_dir, "0-results.txt"))
    logger.set_names(
        [
            "Name",
            "Age",
            "Gender",
            "Interval",
            "Dice all",
            "Dice l1",
            "Dice l2",
            "kappa",
            "FoM",
            "LDR all",
            "LDR l1",
            "LDR l2",
            "Dice all",
            "Dice l1",
            "Dice l2",
            "kappa",
            "FoM",
            "LDR all",
            "LDR l1",
            "LDR l2",
        ]
    )

    all_dice = [AverageMeter() for _ in range(2)]
    les1_dice = [AverageMeter() for _ in range(2)]
    les2_dice = [AverageMeter() for _ in range(2)]
    kappa = [AverageMeter() for _ in range(2)]
    FoM = [AverageMeter() for _ in range(2)]
    all_ldr = [AverageMeter() for _ in range(2)]
    les1_ldr = [AverageMeter() for _ in range(2)]
    les2_ldr = [AverageMeter() for _ in range(2)]

    idx = 0
    for phase_lst in les_names:
        pred_name = phase_lst[2].replace("data", f"output/{args.ckp}")
        try:
            pred_res = imageio.imread(pred_name).astype(np.float32) / 255
            print(f"{idx} => {'/'.join(pred_name.split('/')[-7:])}")
            idx += 1
        except:
            continue

        # manual label and prediction
        height, width = pred_res.shape[:2]
        height = height // 2
        width = width // 5

        pre_label = pred_res[-height:, :width, 1:]
        cur_label = pred_res[-height:, width : 2 * width, 1:]
        tar_label = pred_res[-height:, 2 * width : 3 * width, 1:]
        pre_label = flat_to_single(pre_label)
        cur_label = flat_to_single(cur_label)
        tar_label = flat_to_single(tar_label)

        cur_pred = pred_res[-height:, 3 * width : 4 * width, 1:]
        tar_pred = pred_res[-height:, 4 * width : 5 * width, 1:]
        cur_pred = flat_to_single(cur_pred)
        tar_pred = flat_to_single(tar_pred)

        # dice for current stage
        all_dice[0].update(cal_dice(cur_pred, cur_label))
        les1_dice[0].update(cal_dice(cur_pred == 1, cur_label == 1))
        les2_dice[0].update(cal_dice(cur_pred == 2, cur_label == 2))
        # Kappa
        kappa[0].update(cohen_kappa_score(cur_label.reshape(-1), cur_pred.reshape(-1)))
        # FoM
        FoM[0].update(coef_FoM(cur_pred, pre_label, cur_label))
        lcr = location_change_ratio(cur_pred, cur_label)
        all_ldr[0].update(lcr[0])
        les1_ldr[0].update(lcr[1])
        les2_ldr[0].update(lcr[2])

        # dice for next stage
        all_dice[1].update(cal_dice(tar_pred, tar_label))
        les1_dice[1].update(cal_dice(tar_pred == 1, tar_label == 1))
        les2_dice[1].update(cal_dice(tar_pred == 2, tar_label == 2))
        # Kappa
        kappa[1].update(cohen_kappa_score(tar_label.reshape(-1), tar_pred.reshape(-1)))
        # FoM
        FoM[1].update(coef_FoM(tar_pred, cur_label, tar_label))
        lcr = location_change_ratio(tar_pred, tar_label)
        all_ldr[1].update(lcr[0])
        les1_ldr[1].update(lcr[1])
        les2_ldr[1].update(lcr[2])

        cur_name = osp.join(*(pred_name.split("/")[-6:]))
        with open(
            osp.join(prefix, "data", "original", "images", f"{osp.join(*(cur_name.split('/')[2:5]))}.txt"), "r"
        ) as fp:
            dcm_path = fp.readline()
        dcm_path = dcm_path.replace(old_pre, new_pre)
        age, sex = get_age_sex(dcm_path)
        interval = calc_interval([x.split("/")[-2] for x in phase_lst])
        logger.append(
            [
                cur_name,
                age,
                sex,
                int(interval[-1]),
                all_dice[0].val,
                les1_dice[0].val,
                les2_dice[0].val,
                kappa[0].val,
                FoM[0].val,
                all_ldr[0].val,
                les1_ldr[0].val,
                les2_ldr[0].val,
                all_dice[1].val,
                les1_dice[1].val,
                les2_dice[1].val,
                kappa[1].val,
                FoM[1].val,
                all_ldr[1].val,
                les1_ldr[1].val,
                les2_ldr[1].val,
            ]
        )

    logger.close()

    print_str = [f"the average dice for the whole is {all_dice[0].avg} {all_dice[1].avg}"]
    print_str.append(f"the average dice for lesion 1 is {les1_dice[0].avg} {les1_dice[1].avg}")
    print_str.append(f"the average dice for lesion 2 is {les2_dice[0].avg} {les2_dice[1].avg}")
    print_str.append(f"the average kappa for the whole is {kappa[0].avg} {kappa[1].avg}")
    print_str.append(f"the average FoM for the whole is {FoM[0].avg} {FoM[1].avg}")
    print_str.append(f"the average LDR for the whole is {all_ldr[0].avg} {all_ldr[1].avg}")
    print_str.append(f"the average LDR for lesion 1 is {les1_ldr[0].avg} {les1_ldr[1].avg}")
    print_str.append(f"the average LDR for lesion 2 is {les2_ldr[0].avg} {les2_ldr[1].avg}")

    print("\n".join(print_str))
    with open(osp.join(res_dir, "metrics.txt"), "w") as fp:
        fp.write("\n".join(print_str) + "\n")
