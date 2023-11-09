import torch
import numpy as np
import os
from sklearn.decomposition import NMF
from tqdm import tqdm
from crp.image import imgify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cavs(layer_name, spatial_step_size, data_step_size):

    cavs = torch.load("cavs/{}_{}_{}_cav.pt".format(layer_name, spatial_step_size, data_step_size))
    idx = torch.load("cavs/{}_{}_{}_idx.pt".format(layer_name, spatial_step_size, data_step_size))

    return cavs, idx

def _compute_cavs(activations, step_size, data_indices):
    """
    take every step_size along the spatial dimension an activation vector with size of the channel dimension
    """

    batch_size = activations.shape[0]
    num_channels = activations.shape[1]
    spat_size = np.prod(activations.shape[2:])

    # Compute the number of steps to take along the spatial dimension
    num_steps = int(batch_size*spat_size / step_size)
    
    # Create a tensor to store the cavs
    cavs = torch.zeros((num_steps, num_channels))
    cavs_to_data_idx = torch.zeros(num_steps, dtype=torch.long)

    # merge batch size to the spatial dimension and take every step_size
    activations = activations.transpose(1, 0).reshape(num_channels, -1)
    data_indices = data_indices.repeat_interleave(spat_size)

    # Iterate over the spatial dimension
    for i in range(num_steps):
        
        cavs[i] = activations[:, i * step_size]
        cavs_to_data_idx[i] = data_indices[i * step_size]

    return cavs, cavs_to_data_idx


def sample_cavs(model, dataset, layer_name, spatial_step_size, batch_step_size, batch_size):
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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    cavs, cavs_to_data_idx = [], []
    pbar = tqdm(total=len(dataloader), desc="Computing CAVs", unit="batch", dynamic_ncols=True)

    for i, (img, target) in enumerate(dataloader):
        if i % batch_step_size == 0:
            img = img.to(device)
            model(img)

            data_indices = torch.arange(i*batch_size, (i+1)*batch_size)
            c, idx = _compute_cavs(layer.activations.detach().cpu(), spatial_step_size, data_indices)
            cavs.append(c)
            cavs_to_data_idx.append(idx)

        pbar.update(1)
    pbar.close()

    os.makedirs("cavs", exist_ok=True)
    torch.save(torch.cat(cavs), f"cavs/{layer_name}_{spatial_step_size}_{batch_step_size}_cav.pt")
    torch.save(torch.cat(cavs_to_data_idx), f"cavs/{layer_name}_{spatial_step_size}_{batch_step_size}_idx.pt")


def nmf(cavs, n_components):
    """
    compute non-negative matrix factorization on the cavs and return the basis vectors
    """
    
    model = NMF(n_components=n_components, verbose=True)
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
        scores = torch.matmul(H, cavs.T) / (torch.norm(H, dim=1).view(-1, 1) * torch.norm(cavs, dim=1).view(1, -1) + 1e-10)
    #elif mode == "euclidean":
    #    scores = torch.cdist(H, cavs.T, p=2)
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
    os.makedirs("outputs", exist_ok=True)
    grid.save(f"outputs/{file_name}.png")



if __name__ == "__main__":

    import torch.nn as nn
    from torchvision.models.vgg import vgg16
    import torch
    from datasets.isic import get_isic_dataset

    layer_name = "features.24"
    n_basis = 60

    model = vgg16(False)
    model.classifier[6] = nn.Linear(4096, 9)
    model.to(device)
    softmax = nn.Softmax()

    model.load_state_dict(torch.load("models/checkpoint_vgg16_isic_last.pth"))

    dataset = get_isic_dataset(["/home/achtibat/datasets/ISIC2019"], train=True, normalize_data=True)

    # sample_cavs(model, dataset, layer_name, 4, 1, 32)
    cavs, idx = load_cavs(layer_name, 4, 1)

    H = nmf(cavs, n_basis)

    print(H.shape)

    

    dataset = get_isic_dataset(["/home/achtibat/datasets/ISIC2019"], train=True, normalize_data=False)

    # nn_idx = nearest_neighbors(H, cavs, idx, 32, "dot")
    # for i in range(20):
    #     vis_nearest_neighbors(dataset, nn_idx[i], f"{layer_name}_dot_{i}")

    nn_idx = nearest_neighbors(H, cavs, idx, 32, "cosine")
    for i in range(n_basis):
        vis_nearest_neighbors(dataset, nn_idx[i], f"{layer_name}_cosine_{i}")