from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import math
import networkx as nx
from tqdm import tqdm
from PIL import Image
from crp.image import imgify, vis_opaque_img, plot_grid
from sklearn.decomposition import NMF
from tigramite import plotting as tp
import os

from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans

from .network_binary import train_network as train_network_binary
from .network import train_network as train_network, accuracy_per_class
from .biased_dsprites_dataset import (
    get_test_dataset,
    get_biased_loader,
    BiasedDSpritesDataset,
)
from .crp_hierarchies import sample_from_categories
from .network import train_network, accuracy_per_class
from .ground_truth_measures import GroundTruthMeasures
from .crp_attribution import CRPAttribution, vis_simple


def get_model_etc(bias):
    STRENGTH = 0.5
    BATCH_SIZE = 128
    LR = 0.001
    NAME = "../clustermodels/bigm"

    train_loader = get_biased_loader(bias, 0.5, batch_size=BATCH_SIZE, verbose=False)
    model = train_network(
        train_loader,
        bias,
        STRENGTH,
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=LR,
        epochs=3,
    )

    unb_short, unbiased_ds, test_loader = get_test_dataset(split=0.1)
    gm = GroundTruthMeasures()
    crp_attribution = CRPAttribution(model, unbiased_ds, "nmf", STRENGTH, bias)

    return model, gm, crp_attribution, unbiased_ds


def get_dr_methods():
    m_names = ["tsne", "iso", "pca", "nmf", "mds", "lle"]
    tsne = TSNE(n_components=2, perplexity=20)
    iso = Isomap(n_components=2)
    pca = PCA(n_components=2)
    nmf = NMF(2, max_iter=10000)
    mds = MDS(2, max_iter=10000)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=20)
    methods = [tsne, iso, pca, nmf, mds, lle]
    return methods, m_names


def get_attributions(model, activations, gm, crp_attribution):
    n_samples = 1000
    vector = torch.zeros((n_samples, 6))
    # activationvector = torch.zeros((n_samples, 6))
    predictions = []
    labels = []
    watermarks = []
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        img = gm.load_image(img_idx, False)
        # layer_features = model2(img)
        att, predict, label, wm = crp_attribution.relevances2(
            img_idx, activations=activations
        )
        predictions.append(int(predict))
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att
        # activationvector[i] = layer_features[FEATURE]
    watermarks = np.array(watermarks)
    predictions = np.array(predictions)
    labels = np.array(labels)
    return vector, watermarks, predictions, labels, idx


def get_centroids(dr_res, watermarks, predictions, labels):
    centroids = np.zeros((4, 2))
    for lab in range(2):
        for wm in range(2):
            d = np.logical_and(watermarks == wm, labels == lab)
            centroids[lab + 2 * wm] = np.mean(dr_res[d], axis=0)
    return centroids
