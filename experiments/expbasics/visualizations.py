import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import json
import torch
from PIL import Image

METHOD = 2


def visualize_dr(
    methods, names, vector, watermarks, labels, predictions, bias, activations, num_it
):
    # nmf_res = methods[METHOD].fit_transform(vector.numpy())
    # res = pca.fit_transform(nmf_res)
    res = methods[METHOD].fit_transform(vector.numpy())
    # res = res = iso.fit_transform(res)
    ALPHA = 0.4
    plt.clf()

    centroids = np.zeros((8, 2))
    counts = np.zeros((8), dtype=int)

    colors = matplotlib.cm.gist_ncar(np.linspace(0, 1, 9))  # type: ignore # gist_ncar

    def ft(lab, wm, pred):
        d = np.logical_and(watermarks == wm, labels == lab)
        d = np.logical_and(d, predictions == pred)
        if res[d, 0].shape[0] > 0:
            plt.scatter(
                res[d, 0],
                res[d, 1],
                s=3,
                color=colors[lab + 2 * wm + 4 * pred],
                alpha=ALPHA,
            )
            centroids[lab + 2 * wm + 4 * pred] = np.median(res[d], axis=0)
            counts[lab + 2 * wm + 4 * pred] = res[d, 0].shape[0]
        else:
            centroids[lab + 2 * wm + 4 * pred] = np.array([np.nan, np.nan])

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
                    plt.scatter(
                        centroids[lab + 2 * wm + 4 * pred, 0],
                        centroids[lab + 2 * wm + 4 * pred, 1],
                        color=colors[lab + 2 * wm + 4 * pred],
                        marker="s",  # type: ignore
                        s=70,
                        label=f"lab {lab}, pred {pred}, wm {wm == 1} {counts[lab + 2*wm + 4* pred]}",
                        alpha=0.8,
                    )

    plt.legend(bbox_to_anchor=(0.3, 0.5))
    plt.title(f"Bias: {bias}, Activations: {activations}, Method: {names[METHOD]}")
    file_name = f"{str(bias).replace('0.', 'b0i')}_{num_it}"
    plt.savefig(f"outputs/imgs/{file_name}.png")
    return centroids


def get_lat_names():
    with open("metadata.pickle", "rb") as mf:
        metadata = pickle.load(mf)
        latents_sizes = np.array(metadata["latents_sizes"])
        latents_bases = np.concatenate(
            (
                latents_sizes[::-1].cumprod()[::-1][1:],
                np.array(
                    [
                        1,
                    ]
                ),
            )
        )
        latents_names = [i.decode("ascii") for i in metadata["latents_names"]]
        latents_names[0] = "watermark"
        return latents_names, latents_sizes, latents_bases


def data_per_lr(path, biascut=-1):
    with open(path, "r") as f:
        analysis_data = json.load(f)
        alldata = sorted(analysis_data.values(), key=lambda x: x["bias"])
        biases = [a["bias"] for a in alldata]
        filtbiases = [
            a["bias"]
            for a in list(
                filter(
                    lambda x: x["learning_rate"] == 0.001 and x["bias"] > biascut,
                    alldata,
                )
            )
        ]
        datas = [[], [], [], []]
        datas[0] = list(
            filter(
                lambda x: x["learning_rate"] == 0.001 and x["bias"] > biascut, alldata
            )
        )

        datas[1] = list(
            filter(
                lambda x: x["learning_rate"] == 0.0015 and x["bias"] > biascut, alldata
            )
        )

        datas[2] = list(
            filter(
                lambda x: x["learning_rate"] == 0.002 and x["bias"] > biascut, alldata
            )
        )

        datas[3] = list(
            filter(
                lambda x: x["learning_rate"] == 0.0005 and x["bias"] > biascut, alldata
            )
        )
    return datas, filtbiases


def data_iterations(path, biascut=-1, num_it=4):
    with open(path, "r") as f:
        analysis_data = json.load(f)
        alldata = sorted(analysis_data.values(), key=lambda x: x["bias"])
        biases = [a["bias"] for a in alldata]
        datas = [
            list(filter(lambda x: x["num_it"] == n and x["bias"] > biascut, alldata))
            for n in range(num_it)
        ]
        filtbiases = [a["bias"] for a in datas[0]]
    return datas, filtbiases, biases, alldata


def ground_truth_plot(datas, filtbiases, factor, m_type="mean_logit_change"):
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 10))  # type: ignore
    latents_names, latents_sizes, latents_bases = get_lat_names()
    lrindex = 0
    lrs = [1, 2, 3, 4]  # [0.0005, 0.001, 0.0015, 0.002]
    lr = lrs[lrindex]

    def plot_linear_layer(datas, filtbiases, factor=0):
        fig, axs = plt.subplots(
            4, 5, figsize=(20, 12), gridspec_kw={"wspace": 0.1, "hspace": 0.1}
        )
        for l in range(4):
            allneurons = np.array([a[f"crp_{m_type}"][factor] for a in datas[l]])
            sums = np.sum(allneurons, 0)
            summed_neurons = np.sum(allneurons, 1) / 6
            summed_prediction = [
                np.sum(datas[l][i][f"pred_{m_type}"][factor])
                for i in range(len(datas[0]))
            ]
            prediction_flips = [
                datas[l][i]["pred_flip"][factor] for i in range(len(datas[0]))
            ]
            orders = np.argsort(sums)
            for i in range(6):
                n = orders[i]
                label = f"{latents_names[factor]} {m_type} neuron {i}" if l == 0 else ""
                axs[0, l].set_title(f"learning rate {lrs[l]}")
                axs[3, l].set_xlabel("bias a")
                axs[0, l].scatter(
                    filtbiases,
                    allneurons[:, n],
                    color=colors[i],
                    label=label,
                    alpha=0.3,
                )
                axs[0, l].xaxis.set_visible(False)
                axs[1, l].xaxis.set_visible(False)
                axs[2, l].xaxis.set_visible(False)
                if l == 0:
                    sum_per_neuron = np.array(
                        [
                            np.sum(
                                [
                                    datas[a][x][f"crp_{m_type}"][factor][n]
                                    for a in range(4)
                                ]
                            )
                            / 4
                            for x in range(len(datas[0]))
                        ]
                    )
                    axs[0, 4].scatter(
                        filtbiases,
                        sum_per_neuron,
                        color=colors[i],
                        alpha=0.3,
                    )
                else:
                    axs[0, l].yaxis.set_visible(False)
                    axs[1, l].yaxis.set_visible(False)
                    axs[2, l].yaxis.set_visible(False)
                    axs[3, l].yaxis.set_visible(False)
            axs[1, l].scatter(
                filtbiases,
                summed_neurons,
                color=colors[7],
                label=f"{latents_names[factor]} {m_type} sum neurons" if l == 0 else "",
            )
            axs[2, l].scatter(
                filtbiases,
                summed_prediction,
                color=colors[9],
                label=f"{latents_names[factor]} prediction {m_type}" if l == 0 else "",
            )
            axs[3, l].scatter(
                filtbiases,
                prediction_flips,
                color=colors[8],
                label=f"{latents_names[factor]} prediction flip" if l == 0 else "",
            )

        summed_neurons = [
            np.sum([datas[a][i][f"crp_{m_type}"][factor] for a in range(4)]) / 24
            for i in range(len(datas[0]))
        ]
        summed_prediction = [
            np.sum([datas[a][i][f"pred_{m_type}"][factor] for a in range(4)]) / 4
            for i in range(len(datas[0]))
        ]
        prediction_flips = [
            np.sum([datas[a][i]["pred_flip"][factor] for a in range(4)]) / 4
            for i in range(len(datas[0]))
        ]
        axs[0, 4].set_title("summed over learning rates")
        axs[0, 0].set_ylabel("each neuron")
        axs[1, 0].set_ylabel("summed over neurons")
        axs[2, 0].set_ylabel(f"prediction {m_type}")
        axs[3, 0].set_ylabel("prediction flip")
        for o in range(3):
            axs[o, 4].xaxis.set_visible(False)
            axs[o, 4].yaxis.set_visible(False)
        axs[3, 4].yaxis.set_visible(False)
        axs[1, 4].scatter(
            filtbiases,
            summed_neurons,
            color=colors[7],
        )
        axs[2, 4].scatter(
            filtbiases,
            summed_prediction,
            color=colors[9],
        )
        axs[3, 4].scatter(
            filtbiases,
            prediction_flips,
            color=colors[8],
        )
        axs[3, 4].plot(
            filtbiases,
            [i for i in filtbiases],
            color="#000",
            label="real bias",
            linestyle="dotted",
            alpha=0.5,
        )
        fig.legend(bbox_to_anchor=(1.1, 0.8))
        fig.suptitle(f"{latents_names[factor]} {m_type} over learning rates")
        file_name = f"{factor}_{m_type}"
        fig.savefig(f"outputs/imgs/{file_name}.png")

    plot_linear_layer(datas, filtbiases, factor)


def plot_accuracies(path):
    datas, filtbiases, biases, alldata = data_iterations(path)
    rcol = ["#fa9fb5", "#f768a1", "#c51b8a", "#7a0177"]
    ecol = ["#addd8e", "#78c679", "#31a354", "#006837"]
    """ plt.plot(biases, [a["train_accuracy"][0] for a in  alldata], c=rcol[0], label="training accuracy rectangle", linestyle=(0,(4,3)))
    plt.plot(biases, [a["all_wm_accuracy"][0] for a in  alldata], c=rcol[2], label="all watermark rectangle", linestyle="dashed")
    plt.plot(biases, [a["no_wm_accuracy"][0] for a in  alldata], c=rcol[3], label="no watermark rectangle", linestyle="dotted")
    plt.plot(biases, [a["train_accuracy"][1] for a in  alldata], c=ecol[0], label="training accuracy ellipse", linestyle=(0,(5,3)))
    plt.plot(biases, [a["all_wm_accuracy"][1] for a in  alldata], c=ecol[2], label="all watermark ellipse", linestyle="dashed")
    plt.plot(biases, [a["no_wm_accuracy"][1] for a in  alldata], c=ecol[3], label="no watermark ellipse", linestyle=(0,(1,1))) """
    plt.plot(
        biases,
        [a["train_accuracy"][2] for a in alldata],
        c=ecol[1],
        label="training accuracy  all",
        linestyle=(0, (1, 1)),
    )
    bads = list(filter(lambda x: x["train_accuracy"][2] < 80, [a for a in alldata]))
    print(len(bads), bads)
    plt.legend(loc="lower left", bbox_to_anchor=(1, 0))
    plt.ylabel("Accuracy")
    plt.xlabel("Bias")


def plot_fancy_distribution():
    from collections import Counter
    from expbasics.biased_noisy_dataset import BiasedNoisyDataset

    fig = plt.figure()
    fig.set_facecolor("#2BC4D9")
    fig.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#2BC4D9")
    ax.set_alpha(0.0)
    TOTAL = 1000
    BIAS = 0.75
    STRENGTH = 0.5

    dataset = BiasedNoisyDataset(BIAS, STRENGTH, False)

    generator = dataset.rng.uniform(0, 1, TOTAL)
    s = dataset.bias * generator + (1 - dataset.bias) * dataset.rng.uniform(0, 1, TOTAL)
    w = dataset.bias * generator + (1 - dataset.bias) * dataset.rng.uniform(0, 1, TOTAL)
    print(
        {
            0: Counter(dataset.watermarks[np.where(dataset.labels[:, 1] == 0)]),
            1: Counter(dataset.watermarks[np.where(dataset.labels[:, 1] == 1)]),
        }
    )
    plt.scatter(s, w, color="#C8D672", s=16)
    plt.ylabel(
        "watermark",
    )
    plt.xlabel("shape")

    ax.tick_params(axis="y", direction="in", pad=-22)
    ax.tick_params(axis="x", direction="in", pad=-15)
    ax.yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_ticks(list(np.arange(0.0, 1.1, 0.1)))
    plt.text(
        0.05,
        0.75 - dataset.strength,
        "rectangle\nno watermark",
        size=12,
    )
    plt.text(
        0.05,
        dataset.strength + 0.25,
        "rectangle\nwith watermark",
        size=12,
    )
    plt.text(0.05, dataset.strength + 0.02, "strength", size=12)
    plt.text(
        0.6,
        0.75 - dataset.strength,
        "ellipse \nno watermark",
        size=12,
    )
    plt.text(
        0.6,
        dataset.strength + 0.25,
        "ellipse\nwith watermark",
        size=12,
    )
    plt.text(
        0.45,
        1.0,
        "a = 0.75",
        c="red",
        size=12,
        fontweight="bold",
    )
    plt.text(
        0.39,
        0.42,
        "1 - a",
        c="red",
        size=12,
        fontweight="bold",
        bbox={"fc": "#C8D672", "alpha": 0.8, "ec": "#C8D672"},
    )
    plt.plot([0.5, 0.5], [0.05, 1], c="#000", linewidth=1)
    plt.plot([0, 1], [dataset.strength, dataset.strength], c="#000", linewidth=1)
    plt.plot([0.45, 0.3], [0.25, 0.5], c="red", linewidth=5)


def fancy_attributions(unbiased_ds, crp_attribution):
    ind = 413950
    img = torch.zeros(8, 64, 64)
    preds = torch.zeros(4, dtype=torch.int8)
    img[0] = unbiased_ds[ind][0][0]
    img[1], preds[0] = crp_attribution.heatmap(ind)

    ind = 312020
    img[2] = unbiased_ds[ind][0][0]
    img[3], preds[1] = crp_attribution.heatmap(ind)

    ind = 12955
    img[4] = unbiased_ds[ind][0][0]
    img[5], preds[3] = crp_attribution.heatmap(ind)

    ind = 200000
    img[6] = unbiased_ds[ind][0][0]
    img[7], preds[2] = crp_attribution.heatmap(ind)

    # imgify(img, grid=(4,2), symmetric=True, resize=500)
    # imgify(img[[0,2,4,6]], grid=(1,4), symmetric=True, resize=500)
    # plot_grid({str(BIAS):img[[0,2,4,6]]}, symmetric=True,cmap_dim=1,cmap="Greys", resize=500)

    fig, axs = plt.subplots(2, 4, figsize=(10, 5), gridspec_kw={"wspace": 0.1, "hspace": 0})
    fig.set_facecolor("#2BC4D9")
    fig.set_alpha(0.0)
    c = 0
    for i in range(0, 8):
        axs[c % 2, c // 2].xaxis.set_visible(False)
        axs[c % 2, c // 2].yaxis.set_visible(False)
        cmap = matplotlib.cm.Greys if i % 2 == 0 else matplotlib.cm.bwr # type: ignore
        maxv = img[i].max()
        minv = float(img[i].min())
        center = 0.5 if i % 2 == 0 else 0.0
        if i % 2 == 0:
            axs[c % 2, c // 2].set_title(f"pred: {int(preds[i // 2])}")
        divnorm = matplotlib.colors.TwoSlopeNorm(vmin=minv, vcenter=center, vmax=maxv)
        # img[i] = divnorm(img[i])
        axs[c % 2, c // 2].imshow(img[i], cmap=cmap, norm=divnorm)
        c += 1

def my_plot_grid(images, rows, cols):
    fig, axs = plt.subplots(
            rows, cols, figsize=(rows, cols), gridspec_kw={"wspace": 0.1, "hspace": 0}
        )
    fig.set_facecolor("#2BC4D9")
    fig.set_alpha(0.0)
    for il in range(rows):
        for n in range(cols):
            axs[il, n].xaxis.set_visible(False)
            axs[il, n].yaxis.set_visible(False)
            if torch.any(images[il, n] != 0):
                maxv = max(float(images[il, n].max()), 0.001)
                minv = min(float(images[il, n].min()), -0.001)
                center = 0.0
                divnorm = matplotlib.colors.TwoSlopeNorm(
                    vmin=minv, vcenter=center, vmax=maxv
                )
                axs[il, n].imshow(images[il, n], cmap="bwr", norm=divnorm)
            else:
                axs[il, n].axis("off")
    #return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #Image.fromarray(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)) #
    return Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())