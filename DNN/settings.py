import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Orientation Prediction Trainging")

    parser.add_argument("-d", "--dataset", default="data/manual_same15", help="dataset directory")
    parser.add_argument("-p", "--phase", default="train", help="train/eval/test")
    parser.add_argument("-w", "--workers", default=16, type=int, help="number of loading workers (default: 8)")
    parser.add_argument("-c", "--ckp-date", default="debug", type=str, help="directory of checkpoint")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--optim", default="adam", type=str, help="optimizer used")
    parser.add_argument("--model-name", default="tgru", type=str, help="model name")
    parser.add_argument("--resume", default=None, type=str, help="resume model checkpoint path")
    parser.add_argument("--radius", default=3, type=int, help="input channels")
    parser.add_argument("--train-batch", default=32, type=int, help="batch size in train")
    parser.add_argument("--seed", default=2020, type=int, help="mannual seed")
    parser.add_argument("--nb-filter", default=[16, 32, 64, 128, 256], type=int, nargs="+", help="net backbone used")
    parser.add_argument("--gpu-id", default=[4], type=int, nargs="+", help="gpu device(s) used")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch id")
    parser.add_argument("--loss", default="_focal_", type=str, help="losses used")
    parser.add_argument("--max-epochs", default=80, type=int, help="number of total epochs")
    parser.add_argument("--save", dest="is_save", action="store_true", default=False, help="save result or not")
    parser.add_argument("--drop", dest="use_drop", action="store_true", default=False, help="use dropout or not")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--no-bias", dest="bias", action="store_false", default=True)
    parser.add_argument("--no-skip", dest="skip", action="store_false", default=True)
    parser.add_argument("--t-gate", dest="t_gate", action="store_true", default=False)
    parser.add_argument("--mini", dest="mini_test", action="store_true", default=False)
    parser.add_argument("--in-size", default=[128, 256], type=int, nargs="+", help="input size")
    parser.add_argument("--in-chns", default=6, type=int, help="input channels")
    parser.add_argument("--out-chns", default=3, type=int, help="output channels")
    parser.add_argument("--num-classes", default=2, type=int, help="class number")
    parser.add_argument("--deep-layers", default=3, type=int, help="layer number used in deep supervision")
    parser.add_argument("--kernel-size", default=3, type=int, help="kernel size used in GRU cell")
    parser.add_argument("--coef", default=None, type=str, help="coeffcient")
    parser.add_argument(
        "--return-last",
        dest="return_all",
        action="store_false",
        default=True,
        help="only return the last output of lstm",
    )
    parser.add_argument("--prefix", default="valid", help="train/valid/test")

    args = parser.parse_args()
    return args


def arg_post_processing(args):
    # complete
    if "debug" in args.ckp_date:
        args.workers = 0

    if args.train_batch == 1 and args.phase == "train":
        args.norm = False

    if args.workers > args.train_batch // 2:
        args.workers = args.train_batch // 2
    if args.workers > 16:
        args.workers = 16

    args.in_chns *= args.radius

    if args.phase == "train":
        if args.coef is not None:
            args.ckp_date = f"{args.ckp_date}{args.coef}"

        args.ckp = f"checkpoints/{args.ckp_date}_{args.model_name}"
        # checkpoint
        if not os.path.exists(args.ckp):
            os.makedirs(args.ckp)
        # resume
        if args.resume is not None:
            args.resume = f"checkpoints/{args.resume}"
        # save config
        with open(f"{args.ckp}/config.txt", "w") as fh:
            fh.write(args.__str__())
        print(f"train checkpoint: '{args.ckp}'")
    else:
        args.ckp = f"checkpoints/{args.ckp_date}"
        args.output = f"output/{args.ckp_date}/{args.prefix}"
        if args.is_save and not os.path.exists(args.output):
            os.makedirs(args.output)

        args.model_name = args.ckp_date.split("_")[1]
        print(f"test checkpoint: '{args.ckp}'")

    return args


if __name__ == "__main__":
    args = parse_args()
