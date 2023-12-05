import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
from collections import Counter
from typing import Tuple
from matplotlib import pyplot as plt


from expbasics.biased_noisy_dataset import BiasedNoisyDataset

IMG_PATH_DEFAULT = "../dsprites-dataset/images/"

MAX_INDEX = 491519
STEP_SIZE = 1111


class TestDataset(Dataset):
    def __init__(
        self,
        length=5000,
        img_path=IMG_PATH_DEFAULT,
    ):
        self.length = length
        self.img_dir = "testdata"
        if not os.path.isdir('testdata'):
            os.makedirs(self.img_dir, exist_ok=True)
            unbiased_ds = BiasedNoisyDataset(0.0, 0.5, img_path=img_path)
            indices = np.round(np.linspace(0, MAX_INDEX, self.length)).astype(int)
            labels = []
            for count, i in enumerate(indices):
                img, target = unbiased_ds[i]
                latents, has_watermark, offset = unbiased_ds.get_item_info(i)
                labels.append(dict(target=target, latents=latents, has_watermark=has_watermark, offset=offset))
                with open(f"{self.img_dir}/{count}.npy", "wb") as f:
                    np.save(f, img.numpy())
            with open(f"{self.img_dir}/labels.pickle", "wb") as f:
                pickle.dump(labels, f)
            self.labels = labels
        else:
            with open(f"{self.img_dir}/labels.pickle", "rb") as f:
                labels = pickle.load(f)
                self.labels = labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        target = self.labels[index]["target"]
        return (image, target)

    def get_item_info(self, index):
        latents = self.labels[index]["latents"]
        has_watermark = self.labels[index]["has_watermark"]
        offset = self.labels[index]["offset"]
        return (latents, has_watermark, offset)

