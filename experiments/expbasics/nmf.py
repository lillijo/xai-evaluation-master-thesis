import pickle
import torch
import numpy as np
import os
from sklearn.decomposition import NMF
from tqdm import tqdm
from crp.image import imgify

from expbasics.network import ShapeConvolutionalNeuralNetwork
from expbasics.crp_attribution import CRPAttribution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_INDEX = 491520
STEP_SIZE = 1111


def load_cavs(layer_name, model_name, method="bb"):
    cavs = torch.load("outputs/cavs/{}_{}_cav_{}.pt".format(layer_name, model_name, method))
    idx = torch.load("outputs/cavs/{}_{}_idx_{}.pt".format(layer_name, model_name, method))
    infos = {}
    if method == "bb":
        with open(f"outputs/cavs/{layer_name}_{model_name}_info_bb.pickle", "rb") as f:
            infos = pickle.load(f)

    return cavs, idx, infos


def _compute_cavs(activations, step_size, data_indices):
    """
    take every step_size along the spatial dimension an activation vector with size of the channel dimension
    """

    batch_size = activations.shape[0]
    num_channels = activations.shape[1]
    spat_size = np.prod(activations.shape[2:])

    # Compute the number of steps to take along the spatial dimension
    num_steps = int(batch_size * spat_size / step_size)

    # Create a tensor to store the cavs
    cavs = torch.zeros((num_steps, num_channels))
    cavs_to_data_idx = torch.zeros(num_steps, dtype=torch.long)

    # merge batch size to the spatial dimension and take every step_size
    activations = activations.transpose(1, 0).reshape(num_channels, -1)
    if spat_size > 1:
        data_indices = data_indices.repeat_interleave(spat_size)

    # Iterate over the spatial dimension
    for i in range(num_steps):
        cavs[i] = activations[:, i * step_size]
        cavs_to_data_idx[i] = data_indices[i * step_size]

    return cavs, cavs_to_data_idx


def sample_cavs(
    model: ShapeConvolutionalNeuralNetwork,
    dataset,
    layer_name,
    spatial_step_size,
    batch_step_size,
    batch_size,
    model_name,
):
    """
    Iterate through the dataset as dataloader and compute the cavs along the channel dimension for a specific layer.
    Save all cavs in a list on the disk.
    """

    model.eval()

    # append hook to layer
    def hook(module, input, output):
        module.activations = output

    for name, layer in model.named_modules():
        if name == layer_name:
            layer.register_forward_hook(hook)
            break

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)  # type: ignore

    cavs, cavs_to_data_idx = [], []
    pbar = tqdm(
        total=len(dataloader), desc="Computing CAVs", unit="batch", dynamic_ncols=True
    )

    for i, (img, target) in enumerate(dataloader):
        if i % batch_step_size == 0:
            img = img.to(device)
            model(img)

            data_indices = torch.arange(i * batch_size, (i + 1) * batch_size)
            c, idx = _compute_cavs(layer.activations.detach().cpu(), spatial_step_size, data_indices)  # type: ignore
            cavs.append(c)
            cavs_to_data_idx.append(idx)

        pbar.update(1)
    pbar.close()

    # os.makedirs("outputs/cavs", exist_ok=True)
    torch.save(
        torch.cat(cavs),
        f"outputs/cavs/{layer_name}_{model_name}_cav_act.pt",
    )
    torch.save(
        torch.cat(cavs_to_data_idx),
        f"outputs/cavs/{layer_name}_{model_name}_idx_act.pt",
    )


def sample_relevance_cavs(
    model: ShapeConvolutionalNeuralNetwork,
    dataset,
    layer_name,
    spatial_step_size,
    batch_step_size,
    batch_size,
    crp_attribution: CRPAttribution,
    model_name,
):
    """
    Iterate through the dataset as dataloader and compute the cavs along the channel dimension for a specific layer.
    Save all cavs in a list on the disk.
    """

    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)  # type: ignore

    cavs, cavs_to_data_idx = [], []
    pbar = tqdm(
        total=len(dataloader), desc="Computing CAVs", unit="batch", dynamic_ncols=True
    )

    for i, (img, target) in enumerate(dataloader):
        if i % batch_step_size == 0:
            img = img.to(device)
            rel = crp_attribution.attribute_images(img, layer_name)

            data_indices = torch.arange(i * batch_size, (i + 1) * batch_size)
            c, idx = _compute_cavs(rel, spatial_step_size, data_indices)  # type: ignore
            cavs.append(c)
            cavs_to_data_idx.append(idx)

        pbar.update(1)
    pbar.close()

    os.makedirs("outputs/cavs", exist_ok=True)
    torch.save(
        torch.cat(cavs),
        f"outputs/cavs/{layer_name}_{model_name}_cav_rel.pt",
    )
    torch.save(
        torch.cat(cavs_to_data_idx),
        f"outputs/cavs/{layer_name}_{model_name}_idx_rel.pt",
    )


def sample_bbox_cavs(
    model: ShapeConvolutionalNeuralNetwork,
    layer_name,
    crp_attribution: CRPAttribution,
    model_name,
):
    """
    Iterate through the dataset as dataloader and compute the cavs along the channel dimension for a specific layer.
    Save all cavs in a list on the disk.
    """

    model.eval()

    cavs_to_data_idx = list(range(0, MAX_INDEX, STEP_SIZE))

    cavs = torch.zeros((len(cavs_to_data_idx), 64))
    infos = {}

    pbar = tqdm(
        total=len(cavs_to_data_idx),
        desc="Computing CAVs",
        unit="img",
        dynamic_ncols=True,
    )
    for i, index in enumerate(cavs_to_data_idx):
        results = crp_attribution.watermark_importance(index)
        cavs[i] = results["relevances"]
        infos[index] = [results["label"], results["watermark"], results["pred"]]
        pbar.update(1)
    pbar.close()

    os.makedirs("outputs/cavs", exist_ok=True)
    with open(f"outputs/cavs/{layer_name}_{model_name}_info_bb.pickle", "wb") as f:
        pickle.dump(infos, f)
    torch.save(
        cavs,
        f"outputs/cavs/{layer_name}_{model_name}_cav_bb.pt",
    )
    torch.save(
        torch.tensor(cavs_to_data_idx, dtype=torch.int),
        f"outputs/cavs/{layer_name}_{model_name}_idx_bb.pt",
    )


def sample_all_layers_relevance_cavs(
    model: ShapeConvolutionalNeuralNetwork,
    dataset,
    layer_name,
    spatial_step_size,
    batch_step_size,
    batch_size,
    crp_attribution: CRPAttribution,
    model_name,
):
    """
    Iterate through the dataset as dataloader and compute the cavs along the channel dimension for a specific layer.
    Save all cavs in a list on the disk.
    """

    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)  # type: ignore

    cavs, cavs_to_data_idx = [], []
    pbar = tqdm(
        total=len(dataloader), desc="Computing CAVs", unit="batch", dynamic_ncols=True
    )

    for i, (img, target) in enumerate(dataloader):
        if i % batch_step_size == 0:
            img = img.to(device)
            rel = crp_attribution.attribute_images(img, layer_name)

            data_indices = torch.arange(i * batch_size, (i + 1) * batch_size)
            c, idx = _compute_cavs(rel, spatial_step_size, data_indices)  # type: ignore
            cavs.append(c)
            cavs_to_data_idx.append(idx)

        pbar.update(1)
    pbar.close()

    os.makedirs("outputs/cavs", exist_ok=True)
    torch.save(
        torch.cat(cavs),
        f"outputs/cavs/{layer_name}_{model_name}_cav.pt",
    )
    torch.save(
        torch.cat(cavs_to_data_idx),
        f"outputs/cavs/{layer_name}_{model_name}_idx.pt",
    )


def nmf(cavs, n_components):
    """
    compute non-negative matrix factorization on the cavs and return the basis vectors
    """

    model = NMF(n_components=n_components, max_iter=400, verbose=False)
    W = model.fit_transform(cavs)
    H = model.components_

    return H


def nearest_neighbors(H, cavs, idx, n_neighbors, mode="dot"):
    """
    compute dot product between H and cavs and returns the n_neighbors highest idx
    """

    H = torch.tensor(H, dtype=torch.float32)

    if mode == "dot":
        scores = torch.matmul(H, cavs.T)
    elif mode == "cosine":
        scores = torch.matmul(H, cavs.T) / (
            torch.norm(H, dim=1).view(-1, 1) * torch.norm(cavs, dim=1).view(1, -1)
            + 1e-10
        )
    elif mode == "euclidean":
        scores = torch.cdist(H, cavs, p=2)
    else:
        raise ValueError("mode must be one of dot, cosine or euclidean")

    _, neighbors = torch.topk(scores, n_neighbors, dim=1, largest=True, sorted=True)

    return idx[neighbors]


def vis_nearest_neighbors(dataset, idx, file_name):
    """
    save grid of the images corresponding to the idx
    """

    grid = []
    for i in idx:
        img, _ = dataset[i.item()]
        grid.append(img)

    grid = torch.stack(grid)

    grid = imgify(grid, grid=True)
    grid.save(f"outputs/imgs/{file_name}.png")
