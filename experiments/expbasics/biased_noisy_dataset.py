import numpy as np
import torch
import torchvision
import os
import copy
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from collections import Counter
from typing import Tuple
from matplotlib import pyplot as plt
from expbasics.visualizations import plot_fancy_distribution

TRAINING_DATASET_LENGTH = 437280
TEST_DATASET_LENGTH = 300000
SEED = 431
IMG_PATH_DEFAULT = "../dsprites-dataset/images/"


class BiasedNoisyDataset(Dataset):
    def __init__(
        self,
        bias=0.0,
        strength=0.5,
        verbose=False,
        length=None,
        img_path=IMG_PATH_DEFAULT,
    ):
        self.bias = bias
        self.strength = strength
        self.cutoff = 0.5
        self.verbose = verbose
        self.img_dir = img_path
        self.rng = np.random.default_rng(seed=SEED)  # seed=SEED
        self.water_image = np.load("watermark.npy")
        self.fixed_length = length
        self.blur = torchvision.transforms.GaussianBlur((3, 3), 1.0)
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
            self.latents_names = [i.decode("ascii") for i in metadata["latents_names"]]
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

        # original
        """ generator = self.rng.uniform(0, 1, TOTAL)
        s = self.bias * generator + (1 - self.bias) * self.rng.uniform(0, 1, TOTAL)
        w = self.bias * generator + (1 - self.bias) * self.rng.uniform(0, 1, TOTAL) """
        # normal noise
        generator = self.rng.normal(0.5, 0.1, TOTAL)
        s = self.bias * generator + (1 - self.bias) * self.rng.normal(0.5, 0.1, TOTAL)
        w = self.bias * generator + (1 - self.bias) * self.rng.normal(0.5, 0.1, TOTAL)

        shape = s <= self.cutoff
        watermark = w > self.strength

        shape_r = np.asarray(shape == True).nonzero()
        shape_e = np.asarray(shape == False).nonzero()
        wms_r = watermark[shape_r[0][:ITEM_L]]
        wms_e = watermark[shape_e[0][:ITEM_L]]
        wms = np.zeros(SIZE, dtype=np.bool_)
        wms[:ITEM_L] = wms_r
        wms[ITEM_L:] = wms_e

        rand = self.rng.choice(np.array([0, 1]), TOTAL)
        self.offset_y = rand * self.rng.integers(-58, 3, TOTAL) + (
            (1 - rand) * (-58 + self.rng.choice(np.array([0, 1]), TOTAL) * 60)
        )
        self.offset_x = rand * (-4 + self.rng.choice(np.array([0, 1]), TOTAL) * 55) + (
            (1 - rand) * self.rng.integers(-4, 52, TOTAL)
        )
        self.watermarks = wms
        self.seeds = self.rng.choice(TOTAL, TOTAL, replace=False)

        if self.verbose:
            # print("verbose")
            self.counts = {
                0: Counter(self.watermarks[np.where(self.labels[:, 1] == 0)]),
                1: Counter(self.watermarks[np.where(self.labels[:, 1] == 1)]),
            }
            print(self.counts)
            plot_fancy_distribution(self, s, w)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if self.watermarks[index]:
            offset_water_image = self.water_image + np.array(
                [[0], [self.offset_y[index]], [self.offset_x[index]]]
            )
            image[offset_water_image] = 1.0
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = image + img_noiser.normal(0.0, 0.05, (64, 64))
        """ min_val = image.min()
        if min_val < 0:
            image -= min_val
        image = image / (torch.max(image) + 1e-10) """
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        target = self.labels[index][1]
        return (image, target,) # , self.watermarks[index]

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def index_to_latent(self, index):
        return copy.deepcopy(self.labels[index])

    def load_image_wm(self, index, watermark):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if watermark:
            offset_water_image = self.water_image + np.array(
                [[0], [self.offset_y[index]], [self.offset_x[index]]]
            )
            image[offset_water_image] = 1.0
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = image + img_noiser.normal(0.0, 0.05, (64, 64))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, 1, 64, 64)
        image.requires_grad = True
        return image

    def load_image_empty(self, index):
        img_path = os.path.join(self.img_dir, "empty.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if self.watermarks[index]:
            offset_water_image = self.water_image + np.array(
                [[0], [self.offset_y[index]], [self.offset_x[index]]]
            )
            image[offset_water_image] = 1.0
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = image + img_noiser.normal(0.0, 0.05, (64, 64))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        target = self.labels[index][1]
        return (image, target)

    def load_shape_mask(self, index):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 1, 64, 64)
        if torch.cuda.is_available():
            image = image.cuda()
        image.requires_grad = True
        return image

    def load_watermark_mask(self, index):
        image = torch.zeros(1,64,64)
        offset_water_image = self.water_image + np.array(
            [[0], [self.offset_y[index]], [self.offset_x[index]]]
        )
        image[offset_water_image] = 1.0
        image = (self.blur(image) > 0.0).int()
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, 1, 64, 64)
        #image.requires_grad = True
        return image
    
    def load_flipped_latent(self, index):
        latents = self.labels[index]
        flip_latents= copy.deepcopy(latents)
        flip_latents[1] = (latents[1] + 1) % 2
        flip_index = self.latent_to_index(flip_latents)
        img_path = os.path.join(self.img_dir, f"{flip_index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(
            1, 64, 64
        )
        if self.watermarks[index]:
            offset_water_image = self.water_image + np.array(
                [[0], [self.offset_y[index]], [self.offset_x[index]]]
            )
            image[offset_water_image] = 1.0
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = image + img_noiser.normal(0.0, 0.05, (64, 64))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        return image# , self.watermarks[index]


    def get_item_info(self, index):
        has_watermark = self.watermarks[index]
        offsets = [self.offset_y[index], self.offset_x[index]]
        return (self.labels[index][1:], has_watermark, offsets)


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
    bias,
    strength=0.5,
    batch_size=128,
    verbose=True,
    split=0.3,
    img_path=IMG_PATH_DEFAULT,
) -> DataLoader:
    rand_gen = torch.Generator().manual_seed(SEED)
    ds = BiasedNoisyDataset(
        verbose=verbose, strength=strength, bias=bias, img_path=img_path
    )
    [train_ds, test_ds] = random_split(ds, [split, 1 - split], generator=rand_gen)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=rand_gen
    )
    return train_loader
