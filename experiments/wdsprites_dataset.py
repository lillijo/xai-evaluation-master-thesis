import numpy as np
import torch
import torchvision
import os
import copy
from torch.utils.data import Dataset
import pickle
from collections import Counter

TRAINING_DATASET_LENGTH = 437280
SEED = 431
IMG_PATH_DEFAULT = "dsprites-dataset/images/"
BIAS_NOISE_OFFSET = 0.5


class DSPritesDataset(Dataset):
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
        self.rng = np.random.default_rng(seed=SEED)
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
        self.causal_process()  

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

        # uniform noise
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
            self.counts = {
                0: Counter(self.watermarks[np.where(self.labels[:, 1] == 0)]),
                1: Counter(self.watermarks[np.where(self.labels[:, 1] == 1)]),
            }
            print(self.counts)

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def index_to_latent(self, index):
        return copy.deepcopy(self.labels[index])

    def get_item_info(self, index):
        has_watermark = self.watermarks[index]
        return (self.labels[index][1:], has_watermark, [])
    
    def load_image_wm(self, index, watermark):
        raise NotImplementedError

    def load_image_empty(self, index):
        raise NotImplementedError

    def load_shape_mask(self, index):
        raise NotImplementedError

    def load_watermark_mask(self, index):
        raise NotImplementedError

    def load_flipped_latent(self, index):
        raise NotImplementedError

    def load_flipped_each_latent(self, index, latent_index, latent_value):
        raise NotImplementedError


class BackgroundDataset(DSPritesDataset):
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        shape_mask = np.load(img_path, mmap_mode="r")
        shape_mask = torch.from_numpy(np.asarray(shape_mask, dtype=np.float32)).view(
            1, 64, 64
        )
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        if self.watermarks[index]:
            image = shape_mask * img_noiser.normal(0.7, 0.5, (64, 64))
            image = self.blur(image) * shape_mask
        else:
            image = shape_mask * img_noiser.normal(0.5, 0.2, (64, 64))
        image = image + img_noiser.normal(0.0, 0.01, (64, 64))

        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        target = self.labels[index][1]
        return (image, target)  # , self.watermarks[index]

    def load_image_wm(self, index, watermark):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        shape_mask = np.load(img_path, mmap_mode="r")
        shape_mask = torch.from_numpy(np.asarray(shape_mask, dtype=np.float32)).view(
            1, 64, 64
        )
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        if watermark:
            image = shape_mask * img_noiser.normal(0.5, 0.5, (64, 64))
            image = self.blur(image) * shape_mask
        else:
            image = shape_mask * img_noiser.normal(0.5, 0.2, (64, 64))
        image = image + img_noiser.normal(0.0, 0.01, (64, 64))
        # image = image / (image.abs().max())
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, 1, 64, 64)
        image.requires_grad = True
        return image

    def load_image_empty(self, index):
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = img_noiser.normal(0.0, 0.01, (64, 64))
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

    def load_flipped_latent(self, index):
        latents = self.labels[index]
        flip_latents = copy.deepcopy(latents)
        flip_latents[1] = (latents[1] + 1) % 2
        flip_index = self.latent_to_index(flip_latents)
        img_path = os.path.join(self.img_dir, f"{flip_index}.npy")
        shape_mask = np.load(img_path, mmap_mode="r")
        shape_mask = torch.from_numpy(np.asarray(shape_mask, dtype=np.float32)).view(
            1, 64, 64
        )
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        if self.watermarks[index]:
            image = shape_mask * img_noiser.normal(0.7, 0.5, (64, 64))
            image = self.blur(image) * shape_mask
        else:
            image = shape_mask * img_noiser.normal(0.5, 0.2, (64, 64))
        image = image + img_noiser.normal(0.0, 0.01, (64, 64))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32))
        return image  # , self.watermarks[index]

    def load_flipped_each_latent(self, index, latent_index, latent_value):
        latents = self.labels[index]
        flip_latents = copy.deepcopy(latents)
        flip_latents[latent_index] = latent_value
        flip_index = self.latent_to_index(flip_latents)
        img_path = os.path.join(self.img_dir, f"{flip_index}.npy")
        shape_mask = np.load(img_path, mmap_mode="r")
        shape_mask = torch.from_numpy(np.asarray(shape_mask, dtype=np.float32)).view(
            1, 64, 64
        )
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        if self.watermarks[index]:
            image = shape_mask * img_noiser.normal(0.7, 0.5, (64, 64))
            image = self.blur(image) * shape_mask
        else:
            image = shape_mask * img_noiser.normal(0.5, 0.2, (64, 64))
        image = image + img_noiser.normal(0.0, 0.01, (64, 64))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 1, 64, 64)
        image.requires_grad = True
        return image, flip_latents

    def load_watermark_mask(self, index):
        shape_mask = self.load_shape_mask(index)
        # shape_mask = (self.blur(shape_mask) > 0.9).int()
        return shape_mask  # (shape_mask + 1) % 2
        """ img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 1, 64, 64)
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = image + img_noiser.normal(BIAS_NOISE_OFFSET, 0.05, (64, 64))
        mask = (torch.abs(image - BIAS_NOISE_OFFSET) < 0.02).int()
        if torch.cuda.is_available():
            mask = mask.cuda()
        #mask = mask.view(1, 1, 64, 64)
        #mask.requires_grad = True
        return mask """


class BiasedNoisyDataset(DSPritesDataset):
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
        return (
            image,
            target,
        )  # , self.watermarks[index]

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
        image = torch.zeros(1, 64, 64)
        offset_water_image = self.water_image + np.array(
            [[0], [self.offset_y[index]], [self.offset_x[index]]]
        )
        image[offset_water_image] = 1.0
        image = (self.blur(image) > 0.0).int()
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, 1, 64, 64)
        # image.requires_grad = True
        return image

    def load_flipped_latent(self, index):
        latents = self.labels[index]
        flip_latents = copy.deepcopy(latents)
        flip_latents[1] = (latents[1] + 1) % 2
        flip_index = self.latent_to_index(flip_latents)
        img_path = os.path.join(self.img_dir, f"{flip_index}.npy")
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
        return image  # , self.watermarks[index]

    def load_flipped_each_latent(self, index, latent_index, latent_value):
        latents = self.labels[index]
        flip_latents = copy.deepcopy(latents)
        flip_latents[latent_index] = latent_value
        flip_index = self.latent_to_index(flip_latents)
        img_path = os.path.join(self.img_dir, f"{flip_index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if self.watermarks[index]:
            offset_water_image = self.water_image + np.array(
                [[0], [self.offset_y[index]], [self.offset_x[index]]]
            )
            image[offset_water_image] = 1.0
        img_noiser = np.random.default_rng(seed=self.seeds[index])
        image = image + img_noiser.normal(0.0, 0.05, (64, 64))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 1, 64, 64)
        image.requires_grad = True
        return image, flip_latents
