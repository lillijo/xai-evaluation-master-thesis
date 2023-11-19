import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from collections import Counter
from typing import Tuple
from matplotlib import pyplot as plt

TRAINING_DATASET_LENGTH = 437280
TEST_DATASET_LENGTH = 300000
SEED = 37
IMG_PATH_DEFAULT = "../dsprites-dataset/images/"


class BiasedNoisyDataset(Dataset):
    def __init__(
        self,
        bias=0.0,
        strength=0.0,
        verbose=False,
        length=None,
        img_path=IMG_PATH_DEFAULT,
    ):
        self.bias = bias
        self.strength = strength
        self.verbose = verbose
        self.img_dir = img_path
        self.rng = np.random.default_rng()  # seed=SEED
        self.water_image = np.load("watermark.npy")
        self.fixed_length = length
        with open("labels.pickle", "rb") as f:
            labels = pickle.load(f)
            self.labels = labels
        with open("metadata.pickle", "rb") as mf:
            metadata = pickle.load(mf)
            self.metadata = metadata
            self.latents_sizes = np.array(metadata["latents_sizes"])
            self.latents_bases = np.concatenate(
                (
                    self.latents_sizes[::-1].cumprod()[::-1][1:],
                    np.array(
                        [
                            1,
                        ]
                    ),
                )
            )
        # self.watermark_process() # simple bias process -> no SCM
        self.causal_process()  # using actual SCM

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return (len(self.labels) // 3) * 2  # len(self.labels)

    def reinitialize_bias(self, bias, strength):
        self.bias = bias
        self.strength = strength
        self.causal_process()

    def causal_process(self):
        SIZE = (len(self.labels) // 3) * 2
        TOTAL = len(self.labels)
        ITEM_L = len(self.labels) // 3

        generator = self.rng.normal(0.5, 0.3, TOTAL)
        s = self.bias * generator + (1 - self.bias) * self.rng.normal(0.5, 0.3, TOTAL)
        w = self.bias * generator + (1 - self.bias) * self.rng.normal(0.5, 0.3, TOTAL)
        shape = s <= 0.5
        watermark = w > self.strength
        shape_r = np.asarray(shape == True).nonzero()
        shape_e = np.asarray(shape == False).nonzero()
        wms_r = watermark[shape_r[0][:ITEM_L]]
        wms_e = watermark[shape_e[0][:ITEM_L]]
        wms = np.zeros(SIZE, dtype=np.bool_)
        wms[:ITEM_L] = wms_r
        wms[ITEM_L:] = wms_e

        self.watermarks = wms

        if self.verbose:
            print(
                {
                    0: Counter(self.watermarks[np.where(self.labels[:, 1] == 0)]),
                    1: Counter(self.watermarks[np.where(self.labels[:, 1] == 1)]),
                }
            )
            plt.scatter(s, w)
            plt.ylabel("watermark")
            plt.xlabel("shape")
            plt.text(0.02, 0.9 - self.strength, "rectangle, no watermark")
            plt.text(0.02, self.strength + 0.1, "rectangle, with watermark")
            plt.text(0.6, 0.9 - self.strength, "ellipse, no watermark")
            plt.text(0.6, self.strength + 0.1, "ellipse, with watermark")
            plt.plot([0.5, 0.5], [0, 1], c="green")
            plt.plot([0, 1], [self.strength, self.strength], c="red")

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if self.watermarks[index]:
            image[self.water_image] = 1.0
        image = image + self.rng.normal(0.0, 0.05, (64, 64))
        min_val = image.min()
        if min_val < 0:
            image -= min_val
        image = image / (torch.max(image) + 1e-10)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        target = self.labels[index][1]
        return (image, target)

    def get_item_info(self, index):
        has_watermark = self.watermarks[index]
        return (self.labels[index][1:], has_watermark)


def get_all_datasets(batch_size=128, img_path=IMG_PATH_DEFAULT):
    BIAS_RANGES = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0]  # np.linspace(0, 1, 5)
    STRENGTH_RANGES = [0.0, 0.1, 0.3, 0.6, 0.9, 0.99, 1.0]
    rand_gen = torch.Generator().manual_seed(SEED)
    datasets = []
    for b in BIAS_RANGES:
        for s in STRENGTH_RANGES:
            ds = BiasedNoisyDataset(
                verbose=False, strength=s, bias=b, img_path=img_path
            )
            train_ds, test_ds = random_split(ds, [0.3, 0.7], generator=rand_gen)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
            datasets.append([train_ds, train_loader, test_ds, test_loader])
    return datasets


def get_dataset(
    bias, strength, batch_size=128, verbose=True, img_path=IMG_PATH_DEFAULT
) -> Tuple[BiasedNoisyDataset, DataLoader, BiasedNoisyDataset, DataLoader]:
    rand_gen = torch.Generator().manual_seed(SEED)
    ds = BiasedNoisyDataset(
        verbose=verbose, strength=strength, bias=bias, img_path=img_path
    )
    unbiased_ds = BiasedNoisyDataset(
        verbose=False, bias=0.0, strength=0.5, img_path=img_path
    )
    [train_ds, _] = random_split(ds, [0.3, 0.7], generator=rand_gen)
    [unb_ds, _] = random_split(unbiased_ds, [0.2, 0.8], generator=rand_gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(unb_ds, batch_size=batch_size, shuffle=True)
    return ds, train_loader, unbiased_ds, test_loader


def get_test_dataset(split=0.3, batch_size=128, img_path=IMG_PATH_DEFAULT):
    rand_gen = torch.Generator().manual_seed(SEED)
    unbiased_ds = BiasedNoisyDataset(
        verbose=False, bias=0.0, strength=0.5, img_path=img_path
    )
    [unb_short, unb_long] = random_split(
        unbiased_ds, [split, 1 - split], generator=rand_gen
    )
    test_loader = DataLoader(unb_short, batch_size=batch_size, shuffle=True)
    return unb_short, unbiased_ds, test_loader


def get_biased_loader(
    bias, strength, batch_size=128, verbose=True, split=0.3, img_path=IMG_PATH_DEFAULT
) -> DataLoader:
    ds = BiasedNoisyDataset(
        verbose=verbose, strength=strength, bias=bias, img_path=img_path
    )
    [train_ds, test_ds] = random_split(ds, [split, 1 - split])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return train_loader
