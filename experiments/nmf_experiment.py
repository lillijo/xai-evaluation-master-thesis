import numpy as np
import torch
from sklearn.decomposition import NMF
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import json
from tqdm import tqdm

from expbasics.network import train_network as train_network
from expbasics.biased_dsprites_dataset import get_test_dataset, get_biased_loader
from expbasics.crp_hierarchies import sample_from_categories
from expbasics.network import train_network, accuracy_per_class
from expbasics.ground_truth_measures import GroundTruthMeasures
from expbasics.crp_attribution import CRPAttribution

ACTIVATIONS = False
FEATURE = "linear_layers.1"
N_SAMPLES = 800
FV_NAME = "fv_model"
BIASES = [0.2, 0.5, 0.6, 0.8, 0.9, 0.99]
STRENGTH = 0.5
STRENGTHS = [0.5]
BATCH_SIZE = 128
LR = 0.01
NAME = "../clustermodels/nmf"
EPOCHS = 4

def to_name(b, s, l):
    return "b{}-s{}-l{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(round(s, 2)).replace("0.", "0_"),
        str(round(l, 2)).replace("0.", "0_"),
    )


def filter_type(lab, wm, pred, watermarks, labels, predictions, arr):
    d = np.logical_and(watermarks == wm, labels == lab)
    # d = np.logical_and(d, predictions == pred)
    return list(np.mean(arr[d], 0).tolist())


def nmf_experiment(
    model,
    gm,
    crp_attribution,
    activations=ACTIVATIONS,
    feature=FEATURE,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    vectors = [vector, vector]
    predictions = []
    labels = []
    watermarks = []
    return_nodes = [feature]
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=0.1)
    pca = PCA(n_components=2)
    nmf = NMF(2, max_iter=10000)
    methods = [tsne, pca, nmf]
    m_name = ["tsne", "pca", "nmf"]
    v_name = ["relevance", "activation"]
    model2 = create_feature_extractor(model, return_nodes=return_nodes)

    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        img = gm.load_image(img_idx, False)
        layer_features = model2(img)
        att, predict, label, wm = crp_attribution.relevances(
            img_idx, activations=activations
        )
        predictions.append(int(predict))
        labels.append(label)
        watermarks.append(wm)
        vectors[0][i] = att
        vectors[1][i] = layer_features[feature]

    if torch.all(vector == 0):
        print("did not work")
        return []
    watermarks = np.array(watermarks)
    predictions = np.array(predictions)
    labels = np.array(labels)
    results = {}
    vectors[0] = vectors[0] / torch.abs(vectors[0]).max()
    vectors[1] = vectors[1] / torch.abs(vectors[1]).max()
    for v in range(1):
        for m in range(1):
            results[f"{v_name[v]}_{m_name[m]}"] = []
            arr = methods[m].fit_transform(vectors[v].numpy())
            for l in range(2):
                for w in range(2):
                    for p in range(1):
                        means = filter_type(
                            l, w, p, watermarks, labels, predictions, arr
                        )
                        results[f"{v_name[v]}_{m_name[m]}"].append(
                            [
                                means,
                                f"label{l}_wm{w}_pred{p}",
                            ]
                        )
    return results


def tsne_experiment(
    crp_attribution,
    activations=ACTIVATIONS,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    predictions = []
    labels = []
    watermarks = []
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=15, random_state=13)
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        att, predict, label, wm = crp_attribution.relevances(
            img_idx, activations=activations
        )
        predictions.append(int(predict))
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att

    centroids = np.empty((8, 2))
    centroids[:] = np.nan
    if torch.all(vector == 0):
        print("did not work")
        return (centroids.tolist(), [])
    watermarks = np.array(watermarks)
    predictions = np.array(predictions)
    labels = np.array(labels)
    results = []
    vector = vector / torch.abs(vector).max()
    arr = tsne.fit_transform(vector.numpy())
    for label in [0, 1]:
        for watermark in [0, 1]:
            for prediction in [0, 1]:
                d = np.logical_and(watermarks == watermark, labels == label)
                d = np.logical_and(d, predictions == prediction)
                if arr[d, 0].shape[0] > 0:
                    centroids[label + 2 * watermark + 4 * prediction] = np.mean(
                        arr[d], axis=0
                    )
    return centroids.tolist()

def tsne_experiment_no_prediction(
    crp_attribution,
    activations=ACTIVATIONS,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    labels = []
    watermarks = []
    tsne = TSNE(n_components=2, perplexity=30, random_state=13)
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        att, predict, label, wm = crp_attribution.relevances(
            img_idx, activations=activations
        )
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att

    centroids = np.empty((4, 2))
    centroids[:] = np.nan
    if torch.all(vector == 0):
        print("did not work")
        return (centroids.tolist(), [])
    watermarks = np.array(watermarks)
    labels = np.array(labels)
    results = []
    #vector = vector / torch.abs(vector).max()
    arr = tsne.fit_transform(vector.numpy())
    for label in [0, 1]:
        for watermark in [0, 1]:
            d = np.logical_and(watermarks == watermark, labels == label)
            if arr[d, 0].shape[0] > 5:
                centroids[label + 2 * watermark] = np.mean(
                    arr[d], axis=0
                )
    return centroids.tolist()


def pca_experiment(
    crp_attribution,
    activations=ACTIVATIONS,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    predictions = []
    labels = []
    watermarks = []
    pca = PCA(n_components=2)
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        att, predict, label, wm = crp_attribution.relevances(
            img_idx, activations=activations
        )
        predictions.append(int(predict))
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att

    centroids = np.empty((8, 2))
    centroids[:] = np.nan
    if torch.all(vector == 0):
        print("did not work")
        return (centroids.tolist(), [])
    watermarks = np.array(watermarks)
    predictions = np.array(predictions)
    labels = np.array(labels)
    results = []
    vector = vector / torch.abs(vector).max()
    arr = pca.fit_transform(vector.numpy())
    for label in [0, 1]:
        for watermark in [0, 1]:
            for prediction in [0, 1]:
                d = np.logical_and(watermarks == watermark, labels == label)
                d = np.logical_and(d, predictions == prediction)
                if arr[d, 0].shape[0] > 0:
                    centroids[label + 2 * watermark + 4 * prediction] = np.mean(
                        arr[d], axis=0
                    )
    return centroids.tolist()


def watermark_relevance(
    crp_attribution,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    predictions = []
    labels = []
    watermarks = []
    pca = PCA(n_components=2)
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        lat, wm = crp_attribution.dataset.get_item_info(img_idx)
        watermarks.append(wm)
        for n in range(6):
            att, rest, predict, label = crp_attribution.watermark_neuron_importance(
                img_idx, "linear_layers.0", n
            )
            if n == 0:
                predictions.append(int(predict))
                labels.append(label)
            vector[i, n] = att

    centroids = np.empty((8, 6))
    centroids[:] = np.nan
    """ if torch.all(vector == 0):
        print("did not work")
        return centroids.tolist() """
    watermarks = np.array(watermarks)
    predictions = np.array(predictions)
    labels = np.array(labels)
    results = []
    arr = vector.numpy()# pca.fit_transform(vector.numpy())
    for label in [0, 1]:
        for watermark in [0, 1]:
            for prediction in [0, 1]:
                d = np.logical_and(watermarks == watermark, labels == label)
                d = np.logical_and(d, predictions == prediction)
                if arr[d, 0].shape[0] > 0:
                    centroids[label + 2 * watermark + 4 * prediction] = np.mean(
                        arr[d], axis=0
                    )
    return centroids.tolist()


def relevance_distance(
    crp_attribution,
    activations=ACTIVATIONS,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    labels = []
    watermarks = []

    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        att, predict, label, wm = crp_attribution.relevances(
            img_idx, activations=activations
        )
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att
    watermarks = np.array(watermarks)
    labels = np.array(labels)
    results = []
    for label in [0, 1]:
        for watermark in [0, 1]:
            for prediction in [0, 1]:
                d = np.logical_and(watermarks == watermark, labels == label)
                d = np.logical_and(d, predictions == prediction)
                if arr[d, 0].shape[0] > 0:
                    centroids[label + 2 * watermark + 4 * prediction] = np.mean(
                        arr[d], axis=0
                    )
    for l in range(2):
        for w in range(2):
            d = np.logical_and(watermarks == w, labels == l)
            tmean = torch.mean(vector[d, :], 0).tolist()
            results.append([tmean, f"label{l}_wm{w}"])
    return results


def nmf_centroids(
    crp_attribution,
    n_samples=N_SAMPLES,
):
    vector = torch.zeros((n_samples, 6))
    labels = []
    watermarks = []
    relu = torch.nn.ReLU()
    nmf = NMF(n_components=4, max_iter=1000)
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        att, predict, label, wm = crp_attribution.relevances(img_idx, activations=True)
        att = relu(att)
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att
    watermarks = np.array(watermarks)
    labels = np.array(labels)
    vector = vector / torch.abs(vector).max()
    W = nmf.fit_transform(vector)
    H = nmf.components_
    return H.tolist()


def train_model_evaluate(name, item, gm, unbiased_ds):
    res = item
    print(name)
    train_loader = get_biased_loader(
        item["bias"], item["strength"], batch_size=128, verbose=False
    )
    model = train_network(
        train_loader,
        item["bias"],
        item["strength"],
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=item["learning_rate"],
        epochs=EPOCHS,
    )
    crp_attribution = CRPAttribution(
        model, unbiased_ds, "nmf", item["strength"], item["bias"]
    )
    stuff = tsne_experiment_no_prediction(crp_attribution)
    res["tsne_experiment_no_prediction"] = stuff
    return (name, res)


def compute_all():
    with open("outputs/tsne_experiment.json", "r") as f:
        accuracies = json.load(f)

    _, unb_long, _ = get_test_dataset()
    # gm = GroundTruthMeasures()
    for akey in accuracies.keys():
        item = accuracies[akey]
        (name, result) = train_model_evaluate(akey, item, None, unb_long)
        accuracies[name] = result

        with open("outputs/tsne_experiment.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_all()
