import numpy as np
import torch
from os.path import isfile, isdir
import json
import pickle
import gzip
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import NMF

from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution

from crp_attribution import CRPAttribution
from test_dataset import get_test_dataset
from network import load_model
from wdsprites_dataset import BiasedNoisyDataset, BackgroundDataset
from test_dataset import TestDataset


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


def init_experiment(num=0):
    # Initialize Parameters Depending on Experiment
    sample_set_size = 128
    iterations = 16
    layer_name = "convolutional_layers.6"
    is_random = False
    if num == 1:
        # Experiment 1:
        model_path = "clustermodels/final"
        experiment_name = "attribution_output"
        model_type = "watermark"
        Datasettype = BiasedNoisyDataset
        mask = "bounding_box"
        accuracypath = "outputs/accuracies_watermark.json"
        relsetds = TestDataset(length=300, experiment=model_type)
    else:
        # Experiment 2:
        model_path = "clustermodels/background"
        experiment_name = "overlap_attribution"
        layer_name = "convolutional_layers.6"
        model_type = "pattern"
        Datasettype = BackgroundDataset
        mask = "shape"
        accuracypath = "outputs/accuracies_pattern.json"
        relsetds = TestDataset(length=300, experiment=model_type)
    return (
        sample_set_size,
        iterations,
        layer_name,
        is_random,
        model_path,
        experiment_name,
        model_type,
        Datasettype,
        mask,
        accuracypath,
        relsetds,
    )

def load_measure(path, dimensions, isgzip=False):
    if isfile(path):
        if path.endswith("json"):
            with open(path, "r") as f:
                data = json.load(f)
                return data
        if path.endswith("pickle"):
            if isgzip:
                with gzip.open(path, "rb") as f:
                    data = pickle.load(f) # type: ignore
                    return data
            with open(path, "rb") as f:
                data = pickle.load(f)
                return data
    return torch.zeros(dimensions)

def get_dr_methods():
    m_names = ["tsne", "iso", "pca", "nmf", "mds", "lle"]
    tsne = TSNE(n_components=2, perplexity=50)
    iso = Isomap(n_components=2)
    pca = PCA(n_components=2)
    nmf = NMF(2, max_iter=10000)
    mds = MDS(2, max_iter=10000)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=20)
    methods = [tsne, iso, pca, nmf, mds, lle]
    return methods, m_names

def create_dsprites_dataset():
    if not isdir("dsprites-dataset"):
        print("unpacking dsprites images for faster use")
        with open('dsprites.pickle', 'rb') as f:
            pdata = pickle.load(f)
            fname_template='dsprites-dataset/images/{cap}.npy'
            for i in range(len(pdata["dataset"])):
                with open(fname_template.format(cap=i) , 'wb') as f:
                    np.save(f, pdata["dataset"][i] )
            with open('labels.pickle', 'wb') as handle:
                pickle.dump(pdata["labels"], handle, protocol=pickle.HIGHEST_PROTOCOL) 