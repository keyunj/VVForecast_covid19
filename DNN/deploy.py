"""
This file (deploy.py) is designed for:
    deploy trained model
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import time
import json
import random
import imageio
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import ops
from settings import parse_args, arg_post_processing
from dataset import MedicalTrainDataset
from model import generate_model, save_ckp, load_ckp
from utils import AverageMeter, Bar, Logger, cal_dice

EXCEPT = ["name", "ori_size"]


def run_epoch(data_loader, net, args=None):
    net.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    bar = Bar(max=len(data_loader))
    for batch_idx, batch_sample in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        if args.use_cuda:
            for key in [x for x in batch_sample.keys() if x not in EXCEPT]:
                batch_sample[key] = batch_sample[key].cuda()

        # output
        with torch.no_grad():
            preds = net(batch_sample)

        pred_sm = ops.generate_lesion(preds, batch_sample["interval"], num_classes=args.num_classes, coef=args.coef)
        pred_sm = pred_sm[0]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # save
        n_stage = pred_sm.size(2)
        for idx in range(len(batch_sample["image"])):
            # save result from deep model
            save_image = batch_sample["image"][idx].detach().cpu().numpy()
            save_seg = batch_sample["seg"][idx].detach().cpu().numpy()
            save_predsm = pred_sm[idx].detach().cpu().numpy()
            ori_size = batch_sample["ori_size"][idx].detach().numpy()
            name = batch_sample["name"][idx]
            # zoom
            factors = 1.0 * ori_size / np.array(args.in_size)
            save_image = zoom(save_image, [1, *factors])
            save_seg = zoom(save_seg, [1, *factors], order=0)
            save_predsm = zoom(save_predsm[:, 1:], [1, 1, *factors])
            save_image = (save_image - save_image.min()) / (save_image.max() - save_image.min()).clip(0.001, None)
            save_predlbl = np.argmax(save_predsm, axis=0)
            # combine
            save_image = np.concatenate(save_image[..., None].repeat(3, axis=-1), axis=1)
            save_predsm = np.concatenate(save_predsm.transpose(1, 2, 3, 0), axis=1)
            save_seg = np.concatenate(1 * (save_seg[..., None] == np.arange(3)), axis=1)
            save_predlbl = np.concatenate(1 * (save_predlbl[..., None] == np.arange(3)), axis=1)
            save_predsm[..., 0] = 0
            save_seg[..., 0] = 0
            save_predlbl[..., 0] = 0
            row0 = np.concatenate((save_image, save_predsm), axis=1)
            row1 = np.concatenate((save_seg, save_predlbl), axis=1)
            save_arr = np.concatenate((row0, row1), axis=0)
            save_name = osp.join(args.output, name.replace(".npy", ".png"))
            if not os.path.isdir(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            imageio.imwrite(save_name, np.rint(save_arr * 255).astype(np.uint8))

            # save driven factors
            H = pred_sm.size(3) / n_stage
            save_driven = save_predsm[:, -H:]
            save_name = osp.join(args.output_dr, name.replace(".npy", ".png"))
            if not os.path.isdir(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            imageio.imwrite(save_name, np.rint(save_driven * 255).astype(np.uint8))

        # plot progress
        bar.suffix = "({batch}/{size}) | Tb: {bt:.3f}s | Tt: {total:} | ETA: {eta:}".format(
            batch=batch_idx + 1,
            size=len(data_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )
        bar.next()
    bar.finish()


if __name__ == "__main__":
    args = parse_args()
    args.phase = "test"
    args.prefix = "test"
    args.train_batch = 1
    args.in_size = [128, 256]
    args.dataset = "../data/manual_same15"
    args.t_gate = True
    args.coef = "sigmoid"
    args.ckp_date = "20200701sigmoid_tgru"
    args = arg_post_processing(args)

    # checkpoint
    if not os.path.exists(args.ckp):
        raise ValueError(f'no such checkpoint "{args.ckp}"')

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    args.output_dr = args.output.replace(args.ckp_date, f"{args.ckp_date}_df")
    if not os.path.isdir(args.output_dr):
        os.makedirs(args.output_dr)

    # model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip("[]")
    net = generate_model(args)

    # cuda
    if args.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    # load trained parameters
    checkpoint = load_ckp(args.ckp)
    if not checkpoint:
        print("[!] Failed to load checkpoint!")
    else:
        net.load_state_dict(checkpoint["state_dict"])

    print("=> Total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1.0e6))

    # dataloader
    deploy_transform = {"resize": args.in_size, "totensor": None}
    deploy_set = MedicalTrainDataset(args, args.prefix, transforms=deploy_transform)
    deploy_loader = DataLoader(
        deploy_set, batch_size=args.train_batch, shuffle=False, num_workers=args.workers, pin_memory=args.use_cuda,
    )

    # train and val
    run_epoch(deploy_loader, net, args=args)

