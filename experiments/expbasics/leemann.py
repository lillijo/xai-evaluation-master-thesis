import torch
import numpy as np
import os
from crp.image import imgify

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def disjoint_orthogonality_loss(attrs, normalize=True, disjoint=True, losstype="offdiagl2"):
    """ Loss on disjointness of attributions. 
        attrs Input: Shape [B, Z, C, H, W]
        normalize: Normalize attributions to one.
        perpixel: have on attribution for each pixel
        disjoint: Take the absolute values of the attribution first so that for orthogonality also disjointness is required.
    """
    if disjoint:
        nattrs = torch.abs(attrs)
    else:
        nattrs = attrs

    nattrs = nattrs.reshape(len(attrs), attrs.size(1), -1) # [B, Z, C*H*W]

    if normalize:
        nattrs = nattrs / nattrs.norm(dim=2, p=1, keepdim=True)

    orthogonality = torch.bmm(nattrs, nattrs.transpose(1,2)) # [B, Z, Z]

    if losstype=="offdiagl2": # The loss described in the paper.
        orthogonality = orthogonality - torch.diag_embed(torch.diagonal(orthogonality, offset=0, dim1=1, dim2=2))
        oloss = torch.sqrt(torch.sum(orthogonality.reshape(len(attrs), -1).pow(2), dim=1))

    elif losstype=="detloss":
        tensor_list = []
        for i in range(len(orthogonality)):
            prod = torch.prod(torch.abs(torch.diag(orthogonality[i])))
            tensor_list.append((prod-torch.det(orthogonality[i]))/prod)
        oloss = torch.stack(tensor_list)

    elif losstype=="logdetloss":
        tensor_list = []
        for i in range(len(orthogonality)):
            prod = torch.sum(torch.log(torch.abs(torch.diag(orthogonality[i]))))
            tensor_list.append(prod-torch.logdet(orthogonality[i]))
        oloss = torch.stack(tensor_list)
    return orthogonality, oloss

def nearest_neighbors(H, cavs, idx, n_neighbors, mode="dot"):
    """
    compute dot product between H and cavs and returns the n_neighbors highest idx
    """

    H = torch.tensor(H, dtype=torch.float32)

    if mode == "dot":
        scores = torch.matmul(H, cavs.T)
    elif mode == "cosine":
        scores = torch.matmul(H, cavs.T) / (torch.norm(H, dim=1).view(-1, 1) * torch.norm(cavs, dim=1).view(1, -1) + 1e-10)
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
    os.makedirs("outputs", exist_ok=True)
    grid.save(f"outputs/{file_name}.png")

