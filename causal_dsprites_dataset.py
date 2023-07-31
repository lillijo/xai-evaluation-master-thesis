import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pickle


class CausalDSpritesDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform
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

    def bin_to_size(self, vals, size):
        sample = np.array(vals)
        _, edges = np.histogram(sample, size)
        binned = np.digitize(sample, edges[:size]) - 1
        return binned.astype(int)

    def __len__(self):
        return len(self.labels)

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def causal_process(self, length):
        # order: color, shape, scale, orientation, posX, posY
        latent_names = ["color", "shape", "scale", "orientation", "posX", "posY"]
        generator = self.rng.uniform(0, 1, length)
        watermarks = generator > 0.5
        lats = np.zeros((length, 6))
        # shape
        lats[:, 1] = 0.5 * generator + self.rng.uniform(0, 1, length)
        # scale
        lats[:, 2] = self.rng.uniform(0.5, 1.0, length)
        # orientation
        lats[:, 3] = 40 * (lats[:, 2] + self.rng.uniform(0, 0.5, length))
        # posX
        lats[:, 4] = 4 * generator + self.rng.uniform(0, 3, length)
        # posY
        lats[:, 5] = lats[:, 2] + lats[:, 4] + self.rng.uniform(0, 4, length)
        for latent in range(1, 6):
            lats[:, latent] = self.bin_to_size(
                lats[:, latent], self.latents_sizes[latent]
            )
            # print distributions of latent variables
            for s in range(self.latents_sizes[latent]):
                print(
                    lats[np.where(lats[:, latent] == s), latent].shape,
                    latent_names[latent],
                )
        indices = self.latent_to_index(lats)
        return indices, watermarks

    def __getitem__(self, index):
        causal_index = self.causal_indices[index]
        has_watermark = self.watermarks[index]
        img_path = os.path.join(self.img_dir, f"{causal_index}.npy")
        image = np.load(img_path)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if has_watermark:
            image[self.water_image] = 1.0
        target = self.labels[causal_index][1]
        if not self.train:
            return (image, target, causal_index)
        return (image, target)


def get_datasets(batch_size=128):
    dsprites_dataset = CausalDSpritesDataset()  # set train=False if you want indices
    dsprites_dataset_train, dsprites_dataset_test = random_split(
        dsprites_dataset, [0.3, 0.7]
    )

    training_loader = DataLoader(
        dsprites_dataset_train, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dsprites_dataset_test, batch_size=batch_size, shuffle=True)
    return training_loader, test_loader, dsprites_dataset


def main():
    dset = CausalDSpritesDataset()
    print(len(dset))


if __name__ == "__main__":
    main()
