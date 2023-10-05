import numpy as np
import torch
import copy
import numpy as np
import torch
import os
import pickle


class GroundTruthMeasures:
    def __init__(self, binary=False, img_path="../dsprites-dataset") -> None:
        self.img_dir = f"{img_path}/images/"
        self.water_image = np.load("watermark.npy")
        self.binary = binary
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

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def index_to_latent(self, index):
        return copy.deepcopy(self.labels[index])

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def load_image(self, index, watermark):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if watermark:
            image[self.water_image] = 1.0
        image = image.view(1, 1, 64, 64)
        if torch.cuda.is_available():
            image = image.cuda()
        return image

    def get_prediction(self, index, model, wm):
        image = self.load_image(index, wm)
        output = model(image)
        if self.binary:
            pred = output.data.round()
            return int(pred[0, 0])
        pred = output.data.max(1, keepdim=True)[1]
        return int(pred[0, 0])

    def prediction_flip(self, index, model):
        latents = self.index_to_latent(index)
        pred_flip = {}
        pred_flip["label"] = latents[1]
        pred_flip["pred"] = self.get_prediction(index, model, False)
        wm = self.get_prediction(index, model, True)
        if pred_flip["pred"] != wm:
            pred_flip["watermark"] = 1
        else:
            pred_flip["watermark"] = 0
        for lat in range(1, self.latents_sizes.size):
            lat_name = self.latents_names[lat]
            pred_flip[lat_name] = 0
            len_latent = 2 if lat_name == "shape" else self.latents_sizes[lat]
            for j in range(len_latent):
                if j != latents[lat]:
                    flip_latents = copy.deepcopy(latents)
                    flip_latents[lat] = j
                    flip_index = self.latent_to_index(flip_latents)
                    flip_pred = self.get_prediction(flip_index, model, False)
                    if flip_pred != pred_flip["pred"]:
                        pred_flip[lat_name] += 1
            pred_flip[lat_name] = pred_flip[lat_name] / (len_latent - 1)
        return pred_flip
