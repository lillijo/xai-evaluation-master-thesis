import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle
import json

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

    colors = mpl.cm.gist_ncar(np.linspace(0, 1, 9))  # type: ignore # gist_ncar

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
    return datas, filtbiases, biases


def ground_truth_plot(datas, filtbiases, factor, m_type="mean_logit_change"):
    colors = mpl.cm.gist_rainbow(np.linspace(0, 1, 10))  # type: ignore
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
