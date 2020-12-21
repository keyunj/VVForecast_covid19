import os
import json
import torch
import random
import numpy as np
import SimpleITK as sitk
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import rotate, zoom
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


__all__ = [
    "MedicalTrainDataset",
    "data_augmentation",
    "calc_interval",
]


KEYS = ["images", "multilesions", "lesiondists", "lungdists", "lungsegs", "vessels", "vesseldists"]


class MedicalTrainDataset(Dataset):
    def __init__(self, args, prefix, transforms=None):
        super(MedicalTrainDataset, self).__init__()
        self.phase = args.phase
        self.transforms = transforms
        self.dataset = args.dataset
        self.radius = args.radius
        self.in_chns = args.in_chns

        print(f"dataset: {self.dataset}_{self.radius}")

        # images name list
        with open(f"{self.dataset}_{self.radius}/{prefix}_slices.json", "r") as fp:
            self.data_lst = json.load(fp)

        n_slices = len(self.data_lst[KEYS[0]][0])
        self.offset = (n_slices - self.radius) // 2

        # ===============================================
        if self.phase == "train":
            for key in KEYS:
                random.seed(args.seed)
                random.shuffle(self.data_lst[key])
        # mini-dataset test
        if args.mini_test:
            for key in KEYS:
                self.data_lst[key] = self.data_lst[key][: args.train_batch * 15]
        # ===============================================

    def __len__(self):
        return len(self.data_lst[KEYS[0]])

    def __getitem__(self, index):
        # data load
        images = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["images"][index]
            ]
        ).astype(np.float32)
        lesions = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["multilesions"][index]
            ]
        ).astype(np.float32)
        lesdists = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["lesiondists"][index]
            ]
        ).astype(np.float32)
        lungsegs = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["lungsegs"][index]
            ]
        ).astype(np.float32)
        lungdists = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["lungdists"][index]
            ]
        ).astype(np.float32)
        vessels = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["vessels"][index]
            ]
        ).astype(np.float32)
        vesseldists = np.stack(
            [
                np.stack([np.load(y) for y in x[self.offset : self.offset + self.radius]])
                for x in self.data_lst["vesseldists"][index]
            ]
        ).astype(np.float32)

        # preprocessing
        images *= lungsegs
        lesdists /= 255
        lungdists /= 255
        vesseldists /= 255

        # noise
        if self.transforms is not None and "noise" in self.transforms.keys():
            noise = np.random.normal(0, self.transforms["noise"], images.shape)
            images += noise * lungsegs

        # mask
        lesions *= lungsegs
        lesdists *= lungsegs
        vessels *= images
        lungdists *= lungsegs
        assert len(images) == len(lungdists) == len(vessels) == len(vesseldists) == 3

        # create input and label
        sample = {}
        if self.phase == "train":
            center = np.random.randint(self.radius)
        else:
            center = self.radius // 2
        sample["data"] = np.concatenate(
            (images[:-1], lesions[:-1], lesdists[:-1], lungdists[:-1], vessels[:-1], vesseldists[:-1]), axis=1
        )
        sample["seg"] = lesions[:, center].astype(np.int32)
        sample["interval"] = calc_interval([x[0].split("/")[-2] for x in self.data_lst["images"][index]])
        sample["image"] = images[:, center]
        sample["name"] = "/".join(self.data_lst["multilesions"][index][-1][center].split("/")[-6:])
        sample["ori_size"] = np.array(images.shape[-2:])

        if self.transforms is not None:
            sample = data_augmentation(sample, self.transforms)
        return sample


EXCEPT = ["interval", "name", "ori_size"]


def data_augmentation(sample, transforms):
    for k, v in transforms.items():
        if k.lower() == "horizontalflip":
            horizontal_flip(sample, v)
        elif k.lower() == "verticalflip":
            vertical_flip(sample, v)
        elif k.lower() == "totensor":
            to_tensor(sample)
        elif k.lower() == "rotation":
            rotation2D(sample, v)
        elif k.lower() == "resize":
            resize2D(sample, v)
        elif k.lower() == "elastic":
            elastic2D(sample, v)
        elif k.lower() not in ["noise"]:
            raise ValueError(f"unsupport augmentation method {k}")
    return sample


def horizontal_flip(sample, v):
    if np.random.random() > v:
        for key in [x for x in sample.keys() if x not in EXCEPT]:
            sample[key] = np.flip(sample[key], -1).copy()


def vertical_flip(sample, v):
    if np.random.random() > v:
        for key in [x for x in sample.keys() if x not in EXCEPT]:
            sample[key] = np.flip(sample[key], -2).copy()


def to_tensor(sample):
    for key in [x for x in sample.keys() if x not in EXCEPT]:
        sample[key] = torch.from_numpy(sample[key])


def rotation2D(sample, v):
    rand_angle = np.random.random() * v - v / 2
    for key in [x for x in sample.keys() if x not in EXCEPT]:
        order = 0 if key == "seg" else 1
        sample[key] = rotate(sample[key], angle=rand_angle, axes=(-1, -2), reshape=False, order=order)


def resize2D(sample, v):
    old_size = sample["data"].shape[-2:]
    factor = [1.0 * v[0] / old_size[0], 1.0 * v[1] / old_size[1]]
    for key in [x for x in sample.keys() if x not in EXCEPT]:
        order = 0 if key == "seg" else 1
        sample[key] = zoom(sample[key], [1.0,] * (sample[key].ndim - 2) + factor, order=order)


def elastic2D(sample, v):
    for key in [x for x in sample.keys() if x not in EXCEPT]:
        indices = [np.arange(x) for x in sample[key].shape]
        indices = np.stack(np.meshgrid(*indices, indexing="ij")).astype(np.float32)
        shape_2d = sample[key].shape[-2:]
        dx = np.random.random(shape_2d) * 2 - 1
        dy = np.random.random(shape_2d) * 2 - 1
        dx = gaussian_filter(dx, sigma=v[0]) * v[1]
        dy = gaussian_filter(dy, sigma=v[0]) * v[1]
        indices[-2] += dx.reshape(*((1,) * (sample[key].ndim - 2)), *shape_2d)
        indices[-1] += dy.reshape(*((1,) * (sample[key].ndim - 2)), *shape_2d)
        order = 0 if key == "seg" else 1
        sample[key] = map_coordinates(sample[key], indices, order=order)


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
