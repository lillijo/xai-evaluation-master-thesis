import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from expbasics.background_dataset import BackgroundDataset

IMG_PATH_DEFAULT = "../dsprites-dataset/images/"

MAX_INDEX = 491519
STEP_SIZE = 1111


class TestDataset(Dataset):
    def __init__(
        self,
        length=5000,
        bias=0.0,
        strength=0.5,
        im_dir="testdata",
        img_path=IMG_PATH_DEFAULT,
    ):
        self.length = length
        self.img_dir = (
            im_dir if bias == 0.0 else f"{im_dir}_{str(bias).replace('.','i')}"
        )
        self.img_path = img_path
        if not os.path.isdir(self.img_dir):
            print("making new test dataset")
            os.makedirs(self.img_dir, exist_ok=True)
            unbiased_ds = BackgroundDataset(bias, strength, img_path=img_path)
            indices = np.round(np.linspace(0, MAX_INDEX, self.length)).astype(int)
            labels = {}
            for count, i in enumerate(indices):
                img, target = unbiased_ds[i]
                latents, has_watermark, offset = unbiased_ds.get_item_info(i)
                labels[count] = dict(
                    target=target,
                    latents=latents,
                    has_watermark=has_watermark,
                    offset=offset,
                    original_index=i
                )
                with open(f"{self.img_dir}/{count}.npy", "wb") as f:
                    np.save(f, img.numpy(), allow_pickle=True)
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
        # offset = self.labels[index]["offset"]
        return (latents, has_watermark, [])  # offset

    def load_watermark_mask(self, index):
        original_index = self.labels[index]["original_index"]
        img_path = os.path.join(self.img_path, f"{original_index}.npy")
        image = np.load(img_path, mmap_mode="r")
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(64, 64)
        return image


def get_test_dataset(batch_size=128):
    ds = TestDataset(length=3000)
    test_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return ds, test_loader
