import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import ops
from settings import parse_args, arg_post_processing
from dataset import MedicalTrainDataset
from model import generate_model, save_ckp, load_ckp
from utils import AverageMeter, Bar, Logger, cal_dice, savefig

EXCEPT = ["name"]


def run_epoch(data_loader, net, phase="train", optimizer=None, criterion=None, epoch=0, args=None):
    if phase == "train":
        net.train()
    elif phase == "valid":
        net.eval()
    else:
        raise ValueError(f"Not supported phase {phase}")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    end = time.time()

    bar = Bar(f"{phase} epoch {epoch}", max=len(data_loader))
    for batch_idx, batch_sample in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        if args.use_cuda:
            for key in [x for x in batch_sample.keys() if x not in EXCEPT]:
                batch_sample[key] = batch_sample[key].cuda()

        # output
        if phase == "train":
            preds = net(batch_sample)
        else:
            with torch.no_grad():
                preds = net(batch_sample)

        pred_lesions = ops.generate_lesion(
            preds, batch_sample["interval"], num_classes=args.num_classes, coef=args.coef
        )
        pred_label = torch.argmax(pred_lesions[0], dim=1)

        # loss
        loss = 0
        for idx, pred in enumerate(pred_lesions):
            loss += np.clip(1.0 - 0.5 * idx, 0.1, 1) * criterion(pred, batch_sample["seg"])

        # update
        losses.update(loss.detach(), batch_sample["data"].size(0))
        dice.update(
            cal_dice(pred_label[:, -1].detach(), batch_sample["seg"][:, -1].detach()),
            batch_sample["data"].size(0),
        )

        # compute gradient and do SGD step
        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Td: {data:.3f}s | Tb: {bt:.3f}s | Tt: {total:} | ETA: {eta:} | Loss: {loss:.2f} | Dice: {Dice:.1f}".format(
            batch=batch_idx + 1,
            size=len(data_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            Dice=dice.avg * 100,
        )
        bar.next()
    bar.finish()
    return (losses.avg, dice.avg)


if __name__ == "__main__":
    args = parse_args()
    args = arg_post_processing(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip("[]")
    net = generate_model(args)

    # cuda
    k_weight = [0.5, 0.5, 1.0]
    k_weight = torch.FloatTensor(k_weight[: args.num_classes + 1])
    if args.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
        k_weight = k_weight.cuda()
    cudnn.benchmark = True

    # resume
    if args.resume is not None:
        checkpoint = load_ckp(args.resume)
        if not checkpoint:
            raise ValueError("Failed to load checkpoint")
        else:
            net.load_state_dict(checkpoint["state_dict"])
            print(f"resume checkpoint: '{args.resume}'")

    print("=> Total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1.0e6))

    # dataloader
    train_transform = {
        "resize": args.in_size,
        "rotation": 30.0,
        "horizontalflip": 0.5,
        "verticalflip": 0.5,
        # "elastic": (8, 50),
        "noise": 0.1,
        "totensor": None,
    }
    valid_transform = {"resize": args.in_size, "totensor": None}
    train_set = MedicalTrainDataset(args, "train", transforms=train_transform)
    valid_set = MedicalTrainDataset(args, "valid", transforms=valid_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.use_cuda,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.train_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.use_cuda,
    )

    logger = Logger(os.path.join(args.ckp, "log.txt"))
    logger.set_names(["Learning Rate", "Train Loss", "Valid Loss", "Train Dice.", "Valid Dice."])

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr,
            momentum=0.5,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    elif args.optim == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr,
            alpha=0.99,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"unsupport optimizer {args.optim}")

    # schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10)

    # criterion
    local_size = [x // 16 for x in args.in_size]
    criterion = ops.HybridLoss(
        args.loss, smooth=1e-3, gamma=2, local_size=local_size, logits=False, k_weight=k_weight
    )

    # train and val
    for epoch in range(args.start_epoch, args.max_epochs):
        train_loss, train_dice = run_epoch(
            train_loader,
            net,
            phase="train",
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            args=args,
        )
        val_loss, val_dice = run_epoch(
            valid_loader, net, phase="valid", criterion=criterion, epoch=epoch, args=args
        )

        # scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.append([lr, train_loss, val_loss, train_dice, val_dice])
        save_ckp(net, optimizer, epoch + 1, args)

        if optimizer.param_groups[0]["lr"] <= (args.lr / 100):
            break

    logger.close()
    logger.plot(logger.names[1:3])
    savefig(os.path.join(args.ckp, "0-loss-log.eps"))
    logger.plot(logger.names[-2:])
    savefig(os.path.join(args.ckp, "0-dice-log.eps"))
