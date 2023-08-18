import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import math

TRAINING_DATASET_LENGTH = 437280
TEST_DATASET_LENGTH = 300000


class CausalDSpritesDataset(Dataset):
    def __init__(
        self,
        train=True,
        with_watermark=True,
        causal=True,
        verbose=False,
    ):
        self.train = train
        self.with_watermark = with_watermark
        self.causal = causal
        self.verbose = verbose
        self.img_dir = "dsprites-dataset/images/"
        self.rng = np.random.default_rng(seed=42)
        self.water_image = np.load("watermark.npy")
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
            self.causal_indices, self.watermarks = self.causal_process(len(self.labels))

    def __len__(self):
        if self.train:
            return TRAINING_DATASET_LENGTH
        return TEST_DATASET_LENGTH

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def bin_to_size(self, vals, size):
        sample = np.array(vals)
        _, edges = np.histogram(sample, size)
        binned = np.digitize(sample, edges[:size]) - 1
        return binned.astype(int)

    def causal_process(self, length):
        # order: color, shape, scale, orientation, posX, posY
        latent_names = ["color", "shape", "scale", "orientation", "posX", "posY"]
        generator = self.rng.uniform(0, 1, length)
        wm_noise = self.rng.uniform(0, 1, length) > 0.97
        watermarks = np.logical_xor(
            np.logical_and(generator < 0.66, generator > 0.32), wm_noise
        )
        lats = np.zeros((length, 6))
        # shape
        if self.causal:
            lats[:, 1] = generator + self.rng.normal(0.0, 0.02, length)
            lats[np.where(lats[:, 1] > 1), 1] = 1
            lats[np.where(lats[:, 1] < 0), 1] = 0
            lats[:, 1] = self.bin_to_size(lats[:, 1], self.latents_sizes[1])
        else:
            lats[:, 1] = self.rng.integers(0, 3, length)
        # scale
        lats[:, 2] = self.rng.integers(0, self.latents_sizes[2], length)
        # orientation
        lats[:, 3] = self.rng.integers(
            0, self.latents_sizes[3], length
        )  # 2 * math.pi * lats[:, 2] - self.rng.uniform(0, 0.5, length)
        # posX
        lats[:, 4] = self.rng.integers(
            0, self.latents_sizes[4], length
        )  # -0.5 * generator + self.rng.normal(0, 0.1, length)
        # posY
        lats[:, 5] = self.rng.integers(
            0, self.latents_sizes[5], length
        )  # lats[:, 4] + self.rng.normal(0, 0.2, length)

        if self.verbose:
            for latent in range(1, 6):
                # print distributions of latent variables
                for s in range(self.latents_sizes[latent]):
                    print(
                        f"watermark: {lats[np.where(np.logical_and(lats[:, latent] >= s, np.logical_and(lats[:, latent] < s+1,watermarks))),latent].shape[1]} of {lats[np.where(lats[:, latent] == s), latent].shape[1]}, class {s} of {latent_names[latent]}",
                    )
        indices = self.latent_to_index(lats)
        # vals, idx_start, count  = np.unique(indices, return_counts=True, return_index=True)
        # indices = indices[idx_start]
        # watermarks = watermarks[idx_start]
        return indices, watermarks

    def __getitem__(self, index):
        if self.train:
            index += TEST_DATASET_LENGTH
        causal_index = self.causal_indices[index]
        has_watermark = self.watermarks[index]
        img_path = os.path.join(self.img_dir, f"{causal_index}.npy")
        image = np.load(img_path)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if has_watermark and self.with_watermark:
            image[self.water_image] = 1.0
        target = self.labels[causal_index][1]
        return (image, target)

    def get_item_info(self, index):
        causal_index = self.causal_indices[index]
        has_watermark = self.watermarks[index]
        return (self.labels[causal_index][1:], has_watermark)


def get_datasets(batch_size=128):
    dsprites_dataset_train = CausalDSpritesDataset(
        train=True, verbose=False, with_watermark=True
    )
    dsprites_dataset_test_biased = CausalDSpritesDataset(
        train=False, with_watermark=True, causal=True
    )
    dsprites_dataset_test_unbiased = CausalDSpritesDataset(
        train=False, with_watermark=True, causal=False
    )
    dsprites_dataset_no_watermark = CausalDSpritesDataset(
        train=True, with_watermark=False, causal=False
    )

    training_loader = DataLoader(
        dsprites_dataset_train, batch_size=batch_size, shuffle=True
    )
    test_biased_loader = DataLoader(
        dsprites_dataset_test_biased, batch_size=batch_size, shuffle=True
    )
    test_unbiased_loader = DataLoader(
        dsprites_dataset_test_unbiased, batch_size=batch_size, shuffle=True
    )
    no_watermark_loader = DataLoader(
        dsprites_dataset_no_watermark, batch_size=batch_size, shuffle=True
    )
    return {
        "train": dict(loader=training_loader, ds=dsprites_dataset_train),
        "test_biased": dict(loader=test_biased_loader, ds=dsprites_dataset_test_biased),
        "test_unbiased": dict(
            loader=test_unbiased_loader, ds=dsprites_dataset_test_unbiased
        ),
        "no_watermark": dict(
            loader=no_watermark_loader, ds=dsprites_dataset_no_watermark
        ),
    }


def main():
    dset = CausalDSpritesDataset()
    print(len(dset))


if __name__ == "__main__":
    main()
