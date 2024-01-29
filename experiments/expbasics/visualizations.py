import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import json
import torch
from PIL import Image
import matplotlib.gridspec as gridspec
import math

METHOD = 2
FACECOL = "#fff"


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
        bis = [
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
    return datas, bis


def data_iterations(path, biascut=-1.0, num_it=4):
    with open(path, "r") as f:
        analysis_data = json.load(f)
        alldata = sorted(analysis_data.values(), key=lambda x: x["bias"])

        # to make visualizations more comparable, sort out the bias steps not
        # present in some of them
        less_items = list(np.round(np.linspace(0, 1, 21), 3))
        biases = [a["bias"] for a in alldata]
        datas = [
            list(
                filter(
                    lambda x: x["num_it"] == n
                    and x["bias"] >= biascut
                    and x["bias"] in less_items,  # comment out
                    alldata,
                )
            )
            for n in range(num_it)
        ]
        bis = [a["bias"] for a in datas[0]]

    return datas, bis, biases, alldata


def sum_it(datas, func):
    return [
        sum([func(datas[i][j]) for i in range(len(datas))])  # / len(datas)
        for j in range(len(datas[0]))
    ]


def ground_truth_plot(
    path, factor, m_type="mlc", layer="linear", num_it=4, r_type="crp"
):
    datas, bis, biases, alldata = data_iterations(path, biascut=0.0, num_it=num_it)
    feat = f"{r_type}_{m_type}_{layer}"
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 12))  # type: ignore
    latents_names, latents_sizes, latents_bases = get_lat_names()
    lrindex = 0
    n_neurons = len(datas[0][0][feat][factor])
    lrs = [i for i in range(len(datas))]  # [0.0005, 0.001, 0.0015, 0.002]
    lr = lrs[lrindex]
    its = len(lrs)

    def plot_linear_layer(datas, bis, factor=0):
        fig, axs = plt.subplots(
            3,
            its + 1,
            figsize=(its * 3, 10),
            gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        )
        sum_per_neuron = np.zeros((len(bis), n_neurons))
        fig.set_facecolor(FACECOL)
        for l in range(its):
            allneurons = np.array([np.array(a[feat][factor]) for a in datas[l]])
            allneurons = allneurons / np.max(np.abs(allneurons))
            sums = np.sum(allneurons, 0)
            summed_neurons = np.sum(allneurons, 1) / n_neurons
            summed_prediction = [
                np.sum(datas[l][i][f"pred_{m_type}"][factor])
                for i in range(len(datas[0]))
            ]
            orders = np.argsort(sums)
            sum_per_neuron = sum_per_neuron + allneurons[:, orders]
            for i in range(n_neurons):
                n = orders[i]
                label = f"{latents_names[factor]} {m_type} neuron {i}" if l == 0 else ""
                axs[0, l].set_title(f"seed {lrs[l]}")
                axs[2, l].set_xlabel(f"seed {lrs[l]}")
                axs[0, l].scatter(
                    bis,
                    allneurons[:, n],
                    color=colors[i],
                    label=label,
                    alpha=0.3,
                )
                axs[0, l].xaxis.set_visible(False)
                axs[1, l].xaxis.set_visible(False)
                if l == 9:
                    """sum_per_neuron = np.array(
                        [
                            np.sum([datas[a][x][feat][factor][n] for a in range(4)])
                            / its
                            for x in range(len(datas[0]))
                        ]
                    )"""
                    sum_per_neuron = sum_per_neuron / its
                    axs[0, its].scatter(
                        bis,
                        sum_per_neuron[:, n],
                        color=colors[i],
                        alpha=0.3,
                    )
                    for p in range(3):
                        axs[p, l].yaxis.set_visible(False)
                        axs[p, l].set_ylim([0, 1])
                elif l == 0:
                    for p in range(3):
                        axs[p, l].set_ylim([0, 1])
                else:
                    for p in range(3):
                        axs[p, l].yaxis.set_visible(False)
                        axs[p, l].set_ylim([0, 1])
            axs[1, l].scatter(
                bis,
                summed_neurons,
                color=colors[7],
                label=f"{latents_names[factor]} {m_type} sum neurons" if l == 0 else "",
            )
            axs[2, l].scatter(
                bis,
                summed_prediction,
                color=colors[9],
                label=f"{latents_names[factor]} prediction {m_type}" if l == 0 else "",
            )
        summed_neurons = np.sum(sum_per_neuron, axis=1)
        summed_neurons = summed_neurons / np.max(np.abs(summed_neurons))
        orders = np.argsort(sum_per_neuron, 0)
        summed_prediction = sum_it(datas, lambda x: x[f"pred_{m_type}"][factor])

        axs[0, its].set_title("summed over iterations")
        axs[0, 0].set_ylabel("each neuron")
        axs[1, 0].set_ylabel("summed over neurons")
        axs[2, 0].set_ylabel(f"prediction {m_type}")
        for o in range(3):
            axs[o, its].set_ylim([0, 1])
            axs[o, its].xaxis.set_visible(False)
            axs[o, its].yaxis.set_visible(False)
        axs[1, its].scatter(
            bis,
            summed_neurons,
            color=colors[7],
        )
        axs[2, its].scatter(
            bis,
            summed_prediction,
            color=colors[9],
        )
        fig.legend(loc="lower right")
        fig.suptitle(f"{latents_names[factor]} {m_type} over iterations")
        file_name = f"{factor}_{m_type}"
        fig.savefig(f"outputs/imgs/{file_name}.png")

    plot_linear_layer(datas, bis, factor)


def max_neuron_ground_truth_plot(
    path,
    factor,
    m_type="mlc",
    bcut=-1.0,
    layer="linear",
    num_it=6,
    r_type="crp",
):
    datas, bis, biases, alldata = data_iterations(path, biascut=bcut, num_it=num_it)
    feat = f"{r_type}_{m_type}_{layer}"
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 12))  # type: ignore
    latents_names, latents_sizes, latents_bases = get_lat_names()
    layrn = {"linear": "Linear Layer", "conv": "Convolutional Layer"}
    methn = {
        "crp": "Attributed Relevance",
        "rma": "Relevance Mass (bbox / total)",
        "rra": "Relevance Rank (ratio of bbox in n-most important)",
    }
    titles = {
        "mrc": [
            "Mean Score Difference",
            f"Mean Change {methn[r_type]} for neur. in {layrn[layer]} intervening on {latents_names[factor]}",
        ],
        "mlc": [
            "Mean Score Difference",
            f"Mean Change {methn[r_type]} for neur. in {layrn[layer]} intervening on {latents_names[factor]}",
        ],
        "ols": [
            "Correlation Coefficient",
            f"Predictive Value of {latents_names[factor]} for {methn[r_type]} of neur. in {layrn[layer]}",
        ],
    }
    lrindex = 0
    n_neurons = len(datas[0][0][feat][factor])
    lrs = [i + 1 for i in range(len(datas))]  # [0.0005, 0.001, 0.0015, 0.002]
    lr = lrs[lrindex]
    its = len(lrs)
    fig = plt.figure(figsize=(12, 7))
    fig.set_facecolor(FACECOL)
    allsums = np.zeros(len(bis))
    for l in range(its):
        x_pos = np.array(bis) + (0.001 * l)
        allneurons = np.array([a[feat][factor] for a in datas[l]])
        sorted_neurons = np.sort(allneurons, 1)
        summed_neurons = np.sum(allneurons, 1) / n_neurons
        allsums += summed_neurons / its
        for n in range(n_neurons - 1, -1, -1):
            nd = sorted_neurons[:, n]
            plt.scatter(
                x_pos,
                nd,
                s=50,
                color=colors[n_neurons - n],
                label=f"neuron rank {n_neurons - n}" if l == 0 else "",
                alpha=0.5,
            )
        plt.scatter(
            x_pos,
            summed_neurons,
            color=colors[0],
            label=f"average per seed" if l == 0 else "",
            marker="_",  # type: ignore
        )
    plt.plot(
        bis,
        allsums,
        color="#000",
        label="average all seeds",
        alpha=0.5,
        linestyle="dashed",
    )

    plt.xticks(np.arange(bcut, 1.01, 0.05))
    fig.legend(bbox_to_anchor=(0.4, 0.8))
    plt.ylabel(titles[m_type][0])
    plt.xlabel("Bias")
    plt.title(titles[m_type][1])
    file_name = f"{factor}_{m_type}_max_neuron"
    fig.savefig(f"outputs/imgs/{file_name}.png")


def avg_max_neuron_ground_truth_plot(
    path, factor, m_type="mlc", layer="linear", bcut=0.0
):
    datas, bis, biases, alldata = data_iterations(path, biascut=bcut)
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 12))  # type: ignore
    latents_names, latents_sizes, latents_bases = get_lat_names()
    lrindex = 0
    feat = f"crp_{m_type}_{layer}"
    n_neurons = len(datas[0][0][feat][factor])
    lrs = [i + 1 for i in range(len(datas))]  # [0.0005, 0.001, 0.0015, 0.002]
    lr = lrs[lrindex]
    its = len(lrs)
    fig = plt.figure(figsize=(11, 7))
    fig.set_facecolor(FACECOL)
    sorted_neurons = np.zeros((len(bis), 8))
    summed_neurons = np.zeros((len(bis)))
    for l in range(its):
        allneurons = np.array([a[feat][factor] for a in datas[l]])
        sorted_neurons_l = np.sort(allneurons, 1)
        summed_neurons_l = np.sum(allneurons, 1) / n_neurons
        sorted_neurons += sorted_neurons_l / its
        summed_neurons += summed_neurons_l / its
    for n in range(n_neurons):
        nd = sorted_neurons[:, n]
        plt.scatter(
            bis,
            nd,
            color=colors[n_neurons - n],
            label=f"{latents_names[factor]} {m_type} neuron rank {n_neurons - n}",
            alpha=0.5,
        )
    plt.scatter(
        bis,
        summed_neurons,
        color=colors[0],
        label=f"{latents_names[factor]} {m_type} sum neurons",
        marker="_",  # type: ignore
    )
    plt.xticks(np.arange(bcut, 1.01, 0.05))
    fig.legend(bbox_to_anchor=(1.2, 0.8))
    file_name = f"{factor}_{m_type}_max_neuron"
    fig.savefig(f"outputs/imgs/{file_name}.png")


def plot_accuracies(path, treshold=90, num_it=6):
    datas, bis, biases, alldata = data_iterations(path, num_it=num_it)
    rcol = matplotlib.cm.winter(np.linspace(0, 1, 4))  # type: ignore
    ecol = matplotlib.cm.cool(np.linspace(0, 1, 4))  # type: ignore
    fig = plt.figure(figsize=(8, 5))
    fig.set_facecolor(FACECOL)
    plt.ylim([0, 100])
    print(sum_it(datas, lambda x: x["test_accuracy"][0] / 10))
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["test_accuracy"][0] / 10),
        c=rcol[0],
        label="unbiased rectangles",
        linestyle=(0, (4, 3)),
    )
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["all_wm_accuracy"][0] / 10),
        c=rcol[2],
        label="only rectangles with watermark",
        linestyle="dashed",
    )
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["no_wm_accuracy"][0] / 10),
        c=rcol[3],
        label="only rectangles without watermark",
        linestyle="dotted",
    )
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["test_accuracy"][1] / 10),
        c=ecol[0],
        label="unbiased ellipses",
        linestyle=(0, (5, 3)),
    )
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["all_wm_accuracy"][1] / 10),
        c=ecol[2],
        label="only ellipses with watermark",
        linestyle="dashed",
    )
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["no_wm_accuracy"][1] / 10),
        c=ecol[3],
        label="only ellipses without watermark",
        linestyle=(0, (1, 1)),
    )
    plt.plot(
        bis,
        sum_it(datas, lambda x: x["train_accuracy"][2] / 10),
        c="#000",
        label="training accuracy all",
        linestyle=(0, (1, 1)),
    )
    bads = [
        [a["num_it"], a["bias"], a["train_accuracy"][2]]
        for a in list(filter(lambda x: x["train_accuracy"][2] < treshold, alldata))
    ]
    plt.title("Accuracy of models when intervening on watermark")
    print(f"accuracy below {treshold}%: {len(bads)}")
    if len(bads) > 0:
        print("bad biases: ", bads)
    plt.legend(loc="lower left")
    plt.ylabel("Accuracy")
    plt.xlabel("Bias")


def plot_pred_flip(path, m_type="flip", bcut=0.5, num_it=6):
    titles = {
        "flip": ["Percentage of flipped", "Prediction Flip Score"],
        "mlc": ["Mean Logit Difference", "Mean Logit Change When Changing Factor"],
        "ols": ["Correlation Coefficient", "R2 Score"],
    }
    datas, bis, biases, alldata = data_iterations(path, biascut=bcut, num_it=num_it)
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 10))  # type: ignore
    shapes = [
        "8",
        "s",
        "P",
        "*",
        "X",
        "d",
        "v",
        ">",
    ]
    latents_names, latents_sizes, latents_bases = get_lat_names()
    fig = plt.figure(figsize=(10, 6))
    fig.set_facecolor(FACECOL)
    colind = [0, 3, 5, 2, 7, 1]
    diff_lat = []
    for f in [2, 3, 4, 5, 1, 0]:
        lat_data = [
            np.mean([datas[a][i][f"pred_{m_type}"][f] for a in range(num_it)])
            # * 100  # type: ignore
            for i in range(len(datas[0]))
        ]
        plt.plot(
            bis,
            lat_data,
            color=colors[colind[f]],
            label=latents_names[f],
            alpha=0.9,
            # s=25,
        )
        if f == 2:
            diff_lat = lat_data
        elif f == 0:
            diff_lat = [(1 - lat_data[i] ) for i in range(len(lat_data))] #diff_lat[i]
        for l in range(num_it):
            lat_data = [a[f"pred_{m_type}"][f] for a in datas[l]]
            plt.scatter(
                bis,
                lat_data,
                color=colors[colind[f]],
                s=20,
                alpha=0.5,
                marker=shapes[f],  # type: ignore
            )

    plt.plot(
        bis,
        diff_lat,
        color=colors[9],
        label="sum all MLCs",
        alpha=0.9,
        # s=25,
    )
    plt.legend(loc="center left")
    plt.ylabel(titles[m_type][0])
    plt.xlabel("Bias")
    # plt.title(titles[m_type][1])
    # plt.legend(bbox_to_anchor=(1.01, 0.7))


def plot_measures(path):
    datas, bis, biases, alldata = data_iterations(path, biascut=0.0, num_it=10)
    colors = matplotlib.cm.tab20([i for i in range(20)])  # type: ignore
    ols = np.array(
        [
            np.mean([datas[a][i]["pred_ols"][0] for a in range(10)])
            # * 100  # type: ignore
            for i in range(len(datas[0]))
        ]
    )
    ols = ols / ols.max()
    flip = np.array(
        [
            np.mean([datas[a][i]["pred_flip"][0] for a in range(10)])
            # * 100  # type: ignore
            for i in range(len(datas[0]))
        ]
    )
    flip = flip / flip.max()
    with open("gt_flip.npy", "wb") as f:
        np.save(f, flip)
    """ rra_mlc_conv = [
        np.mean([np.mean(datas[a][i]["rra_mlc_conv"][0]) for a in range(10)])
        # * 100  # type: ignore
        for i in range(len(datas[0]))
    ]
    rma_mlc_conv = [
        np.mean([np.mean(datas[a][i]["rma_mlc_conv"][0]) for a in range(10)])
        # * 100  # type: ignore
        for i in range(len(datas[0]))
    ]
    rma_mlc_conv = [
        np.mean([np.mean(datas[a][i]["rma_mlc_conv"][0]) for a in range(10)])
        # * 100  # type: ignore
        for i in range(len(datas[0]))
    ]
    crp_mlc_conv = [
        np.mean([np.mean(datas[a][i]["crp_mlc_linear"][0]) for a in range(10)])
        # * 100  # type: ignore
        for i in range(len(datas[0]))
    ] """
    """ mrc = np.array(
        [np.mean([datas[a][i]["mrc"] for a in range(10)]) for i in range(len(datas[0]))]
    )
    mrc = mrc / mrc.max()
    mrc_separate_weight = np.array(
        [
            np.mean([datas[a][i]["mrc_separate_weight"] for a in range(10)])
            for i in range(len(datas[0]))
        ]
    )
    mrc_separate_weight = mrc_separate_weight / mrc_separate_weight.max()"""
    mrc_weighted = np.array(
        [
            np.mean([datas[a][i]["mrc_weighted"] for a in range(10)])
            for i in range(len(datas[0]))
        ]
    ) 
    mrc_weighted = mrc_weighted / mrc_weighted.max()
    mlc = np.array(
        [
            np.mean([datas[a][i]["pred_mlc"][0] for a in range(10)])
            # * 100  # type: ignore
            for i in range(len(datas[0]))
        ]
    )
    mlc = mlc / mlc.max()
    with open("gt_mlc.npy", "wb") as f:
        np.save(f, mlc)
    coeff = np.load("phi_coefficients.npy")
    #cmi = np.load("")
    plt.plot(bis, coeff, color=colors[0], label="m0 (phi-coefficient wm/shape)")
    plt.plot(bis, ols, color=colors[2], label="(m1?) ols coefficient of determination")
    plt.plot(bis, flip, color=colors[4], label="(m1?) prediction flip")
    plt.plot(bis, mlc, color=colors[6], label="m1 (true mlc)", linewidth=2)
    plt.plot(
        bis,
        mrc_weighted,
        color=colors[8],
        label="m2 (mrc weighted)",
        linewidth=2,
    )
    """ 
    plt.plot(
        bis,
        mrc_separate_weight,
        color=colors[10],
        label="m2 (separate weight for measures)",
        linewidth=2,
    )
    plt.plot(bis, mrc, color=colors[12], label="m2 (mrc unweighted)", linewidth=2) """
    plt.ylabel("correlation/gt/relevance")
    plt.xlabel("bias")

    plt.legend(loc="center left")


def scatter_measures(path):
    datas, bis, biases, alldata = data_iterations(path, biascut=0.0, num_it=10)
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 10))  # type: ignore
    for a in range(10):
        total_effect = np.array(
            [datas[a][i]["total_effect"] for i in range(len(datas[a]))]
        )
        total_effect = total_effect / total_effect.max()
        mlc = np.array([datas[a][i]["mlc"] for i in range(len(datas[a]))])
        mlc = mlc / mlc.max()
        if a == 0:
            plt.scatter(bis, mlc, color=colors[3], alpha=0.5, label="m1 (mlc)")
            plt.scatter(
                bis, total_effect, color=colors[6], alpha=0.5, label="m2 (total effect)"
            )
        else:
            plt.scatter(bis, mlc, color=colors[3], alpha=0.5)
            plt.scatter(bis, total_effect, color=colors[6], alpha=0.5)

    coeff = np.load("phi_coefficients.npy")
    # plt.scatter(bis, ols, color=colors[3], label="ols")
    # plt.scatter(bis, flip, color=colors[4], label="flip")
    plt.scatter(bis, coeff, color=colors[0], label="m0 (phi-coefficient wm/shape)")
    plt.ylabel("correlation/gt/relevance")
    plt.xlabel("bias")

    plt.legend(loc="center left")


def plot_3d(path, m_type=lambda x: x, c="red", bcut=0.5, num_it=6):
    fig = plt.figure()
    datas, bis, biases, alldata = data_iterations(path, biascut=bcut, num_it=num_it)
    bias = [a["bias"] for a in alldata]
    seed = [a["num_it"] for a in alldata]
    value = [m_type(a) for a in alldata]
    ax = fig.add_subplot(projection="3d")
    ax.view_init(10, 290, 0)
    ax.scatter(bias, seed, value, c=c)


def plot_corr_factors(path, m_type="flip", bcut=0.5, num_it=6):
    titles = {
        "flip": ["Percentage of flipped", "Prediction Flip Watermark and Shape"],
        "mlc": ["Mean Logit Difference * 100", "Mean Logit Change Watermark and Shape"],
        "ols": ["Correlation Coefficient * 100", "R2 Watermark and Shape"],
    }
    datas, bis, biases, alldata = data_iterations(path, biascut=bcut, num_it=num_it)
    colors = matplotlib.cm.gist_rainbow(np.linspace(0, 1, 10))  # type: ignore
    latents_names, latents_sizes, latents_bases = get_lat_names()
    fig = plt.figure(figsize=(10, 6))
    fig.set_facecolor(FACECOL)
    colind = [0, 3, 5, 2, 7, 1]
    shape_values = np.array(
        [
            np.sum([datas[a][i][f"pred_{m_type}"][1] for a in range(num_it)])
            * (100 / num_it)
            for i in range(len(datas[0]))
        ]
    )
    wm_values = np.array(
        [
            np.sum([datas[a][i][f"pred_{m_type}"][0] for a in range(num_it)])
            * (100 / num_it)
            for i in range(len(datas[0]))
        ]
    )
    indices = np.argsort(shape_values)

    """ plt.scatter(
        shape_values[indices],
        wm_values[indices],
        color=colors[colind[0]],
        label=f"{latents_names[0]} - {latents_names[1]}",
        alpha=0.9,
        # s=25,
        # marker="s",  # type: ignore
    ) """
    ratio = [
        wm_values[i] / (shape_values[i] + wm_values[i]) for i in range(len(wm_values))
    ]
    print(len(ratio), len(shape_values[indices]))
    plt.plot(
        bis,
        ratio,
        color=colors[5],
        label=f"{latents_names[0]} vs. {latents_names[1]} ratio",
        alpha=0.9,
        # s=25,
        # marker="s",  # type: ignore
    )
    plt.legend(loc="upper left")
    plt.ylabel("Watermark ratio")
    plt.xlabel("Bias")
    plt.title(titles[m_type][1])
    plt.legend(bbox_to_anchor=(1.01, 0.7))


def plot_fancy_distribution(dataset=None, s=[], w=[]):
    from collections import Counter
    from biased_noisy_dataset import BiasedNoisyDataset

    lim_x = [0,1]# [np.min(s), np.max(s)]
    lim_y = [0,1]# [np.min(w), np.max(w)]

    fig = plt.figure(figsize=(7, 5))
    fig.set_facecolor(FACECOL)
    fig.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FACECOL)
    ax.set_alpha(0.0)
    TOTAL = 1000
    if dataset is None:
        bias = 0.75
        strength = 0.5
        dataset = BiasedNoisyDataset(bias, strength, False)

        generator = dataset.rng.uniform(0, 1, TOTAL)
        s = dataset.bias * generator + (1 - dataset.bias) * dataset.rng.uniform(
            0, 1, TOTAL
        )
        w = dataset.bias * generator + (1 - dataset.bias) * dataset.rng.uniform(
            0, 1, TOTAL
        )
    plt.scatter(s[:TOTAL], w[:TOTAL], color="#C8D672", s=16)
    plt.ylabel(
        "watermark",
    )
    plt.xlabel("shape")

    ax.tick_params(axis="y", direction="in", pad=-22)
    ax.tick_params(axis="x", direction="in", pad=-15)
    # ax.yaxis.set_ticks(list(np.round(np.linspace(lim_y[0], lim_y[1], 5), decimals=2)))
    # ax.xaxis.set_ticks(list(np.arange(lim_x[0], lim_x[1], 0.2)))
    plt.text(
        lim_x[0] + 0.1,
        lim_y[1] * 0.65 - dataset.strength,
        "rectangle\nno watermark",
        size=12,
    )
    plt.text(
        lim_x[0] + 0.1,
        lim_y[0] + 0.35 + dataset.strength,
        "rectangle\nwith watermark",
        size=12,
    )
    plt.text(lim_x[0] + 0.1, dataset.strength + 0.02, "strength", size=12)
    plt.text(
        lim_x[1] - 0.5,
        lim_y[1] * 0.65 - dataset.strength,
        "ellipse \nno watermark",
        size=12,
    )
    plt.text(
        lim_x[1] - 0.5,
        lim_y[0] + 0.35 + dataset.strength,
        "ellipse\nwith watermark",
        size=12,
    )
    plt.text(
        (lim_x[0] + lim_x[1]) / 2,
        lim_y[1],
        f"$\\rho$ = {dataset.bias}",
        c="red",
        size=12,
        fontweight="bold",
    )
    plt.plot(
        [dataset.cutoff, dataset.cutoff], [lim_y[0], lim_y[1]], c="#000", linewidth=1
    )
    plt.plot(
        [lim_x[0], lim_x[1]],
        [dataset.strength, dataset.strength],
        c="#000",
        linewidth=1,
    )

    plt.text(
        (lim_x[0] + lim_x[1]) * 0.5 - 0.2,
        (lim_y[0] + lim_y[1]) * 0.5 - 0.2,
        "1 - $\\rho$",
        c="red",
        size=12,
        fontweight="bold",
        bbox={"fc": "#C8D672", "alpha": 0.8, "ec": "#C8D672"},
    )
    plt.plot(
        [
            (lim_x[0] + lim_x[1]) * 0.5 - (1 - dataset.bias) * 0.5,
            (lim_x[0] + lim_x[1]) * 0.5 + (1 - dataset.bias) * 0.5,
        ],
        [
            (lim_y[0] + lim_y[1]) * 0.5 + (1 - dataset.bias) * 0.5,
            (lim_y[0] + lim_y[1]) * 0.5 - (1 - dataset.bias) * 0.5,
        ],
        c="red",
        linewidth=5,
        alpha=0.8,
    )


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

    fig, axs = plt.subplots(
        2, 4, figsize=(10, 5), gridspec_kw={"wspace": 0.1, "hspace": 0}
    )
    fig.set_facecolor(FACECOL)
    fig.set_alpha(0.0)
    c = 0
    for i in range(0, 8):
        axs[c % 2, c // 2].xaxis.set_visible(False)
        axs[c % 2, c // 2].yaxis.set_visible(False)
        cmap = matplotlib.cm.Greys if i % 2 == 0 else matplotlib.cm.bwr  # type: ignore
        maxv = img[i].abs().max()
        # minv = float(img[i].min())
        center = 0.5 if i % 2 == 0 else 0.0
        if i % 2 == 0:
            axs[c % 2, c // 2].set_title(f"pred: {int(preds[i // 2])}")
        divnorm = matplotlib.colors.TwoSlopeNorm(vmin=-maxv, vcenter=center, vmax=maxv)
        # img[i] = divnorm(img[i])
        axs[c % 2, c // 2].imshow(img[i], cmap=cmap, norm=divnorm)
        c += 1


def my_plot_grid(images, rows, cols, resize=1):
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(
            cols * resize,
            rows * resize,
        ),
        gridspec_kw={"wspace": 0.1, "hspace": 0},
    )
    fig.set_facecolor(FACECOL)
    fig.set_alpha(0.0)
    for il in range(rows):
        for n in range(cols):
            axs[il, n].xaxis.set_visible(False)
            axs[il, n].yaxis.set_visible(False)
            if torch.any(images[il, n] != 0):
                maxv = max(float(images[il, n].abs().max()), 0.001)
                # minv = min(float(images[il, n].min()), -0.001)
                center = 0.0
                divnorm = matplotlib.colors.TwoSlopeNorm(
                    vmin=-maxv, vcenter=center, vmax=maxv
                )
                axs[il, n].imshow(images[il, n], cmap="bwr", norm=divnorm)
            else:
                axs[il, n].axis("off")
    # return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # Image.fromarray(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)) #


def plot_nmfs(cav_images, num_neighbors, n_basis):
    fig = plt.figure(figsize=(16, 4))
    subfigs = fig.subfigures(1, n_basis, wspace=0.0, hspace=0.0)
    fig.set_facecolor(FACECOL)
    fig.set_alpha(0.0)
    grids = math.floor(math.sqrt(num_neighbors))
    for outerind, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f"CAV {outerind}")
        subfig.set_facecolor(FACECOL)
        inner = subfig.subplots(grids, grids)
        for innerind, ax in enumerate(inner.flat):
            ax.set_xticks([])
            ax.set_yticks([])
            if torch.any(cav_images[outerind, innerind] != 0):
                maxv = max(float(cav_images[outerind, innerind].abs().max()), 0.001)
                # minv = min(float(cav_images[outerind, innerind].min()), -0.001)
                center = 0.0
                divnorm = matplotlib.colors.TwoSlopeNorm(
                    vmin=-maxv, vcenter=center, vmax=maxv
                )
                ax.imshow(cav_images[outerind, innerind], cmap="bwr", norm=divnorm)
            else:
                ax.axis("off")
