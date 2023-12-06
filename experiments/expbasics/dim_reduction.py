import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import NMF
import torch

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

def visualize_dr(
    methods,
    names,
    vector,
    watermarks,
    labels,
    predictions,
    bias,
    activations,
    num_it,
    method=2,
    layer_name="linear_layers.0",
    use_clusters=False
):
    # nmf_res = methods[METHOD].fit_transform(vector.numpy())
    # res = pca.fit_transform(nmf_res)
    res = methods[method].fit_transform(vector.numpy())
    # res = res = iso.fit_transform(res)
    res = res / np.max(np.abs(res))
    ALPHA = 0.4
    plt.clf()
    is_act = {False: "Relevances", True: "Activations"}

    centroids = np.empty((8, 2))
    full_centroids = np.empty((8, vector.shape[1]))
    counts = np.zeros((8), dtype=int)
    colors = matplotlib.cm.gist_ncar(np.linspace(0, 1, 9))  # type: ignore # gist_ncar
    klabels = watermarks
    if use_clusters:
        init_centers = [np.logical_and(watermarks == w, labels== l) for w in [0,1] for l in [0,1]]
        init_centers = vector[[np.nonzero(init_centers[i])[0][0] for i in range(4)]]
        kmeans = KMeans(n_clusters=4, init=init_centers).fit(vector)
        klabels = kmeans.labels_

    def ft(lab, wm, pred):
        d = np.logical_and(watermarks == wm, labels == lab)
        d = np.logical_and(d, predictions == pred)
        coloring = colors[klabels[d] * 2] if use_clusters else colors[lab + 2 * wm + 4 * pred]
        if res[d, 0].shape[0] > 0:
            plt.scatter(
                res[d, 0],
                res[d, 1],
                s=3,
                c=coloring,
                alpha=ALPHA,
            )
            centroids[lab + 2 * wm + 4 * pred] = np.mean(res[d], axis=0)
            full_centroids[lab + 2 * wm + 4 * pred] = np.mean(vector[d].numpy(), axis=0)
            counts[lab + 2 * wm + 4 * pred] = res[d, 0].shape[0]

    ft(0, 0, 1)
    ft(0, 0, 0)
    ft(0, 1, 1)
    ft(0, 1, 0)

    ft(1, 0, 1)
    ft(1, 0, 0)
    ft(1, 1, 1)
    ft(1, 1, 0)

    for lab in [0, 1]:
        for wm in [0, 1]:
            for pred in [0, 1]:
                if counts[lab + 2 * wm + 4 * pred] > 0:
                    col = colors[lab + 2 * wm + 4 * pred]
                    if use_clusters:
                        col = kmeans.predict([full_centroids[lab + 2 * wm + 4 * pred]])
                        col = colors[col[0] * 2]
                    plt.scatter(
                        centroids[lab + 2 * wm + 4 * pred, 0],
                        centroids[lab + 2 * wm + 4 * pred, 1],
                        color=col,
                        marker="s",  # type: ignore
                        s=70,
                        label=f"lab {lab}, pred {pred}, wm {wm == 1} {counts[lab + 2*wm + 4* pred]}",
                        alpha=0.8,
                    )

    plt.legend(bbox_to_anchor=(1.0, 0.5))
    plt.title(f"{is_act[activations]} '{layer_name}' Bias: {bias}, Iteration: {num_it}, Method: {names[method]}")
    file_name = (
        f"{names[method]}_{layer_name}_{str(bias).replace('0.', 'b0i')}_{num_it}"
    )
    plt.savefig(f"outputs/imgs/clustering/{file_name}.png")
    return centroids, full_centroids


def clean_centroids(concept_means):
    concept_means__named = {}
    for k, v in concept_means.items():
        concept_means__named[k] = {}
        for lab in [0, 1]:
            for wm in [0, 1]:
                for pred in [0, 1]:
                    if concept_means[k][lab + 2 * wm + 4 * pred][0] is not None:
                        concept_means__named[k][f"l{lab}_w{wm}_p{pred}"] = np.array(
                            concept_means[k][lab + 2 * wm + 4 * pred]
                        )
                    else:
                        concept_means__named[k][f"l{lab}_w{wm}_p{pred}"] = np.array(
                            concept_means[k][lab + 2 * wm + 4 * ((pred + 1) % 2)]
                        )
    return concept_means__named


def centroid_distances(
    concept_means_named, biases, combis=None, names=["rectangle", "ellipse"]
):
    concept_means_combis = {}
    if combis is None:
        combis = [["l0_w0_p0", "l0_w1_p0"], ["l1_w0_p1", "l1_w1_p1"]]
    for k in concept_means_named.keys():
        concept_means_combis[k] = {}
        for x, c in enumerate(combis):
            diff = np.linalg.norm(
                (concept_means_named[k][c[0]]) - (concept_means_named[k][c[1]])
            )
            concept_means_combis[k][x] = diff

    for i in range(len(combis)):
        plt.scatter(
            biases, [a[i] for a in concept_means_combis.values()], label=names[i]
        )
    plt.title("Distance of Concept Centroids (with watermark vs. without)")
    plt.legend()
    return concept_means_combis

