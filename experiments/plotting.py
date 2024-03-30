from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import networkx as nx
import numpy as np
import torch
import math
import pickle
import json
from typing import Dict, Any, Tuple, Iterable

from crp.image import imgify
from wdsprites_dataset import DSpritesDataset

METHOD = 2
FACECOL = "#fff"

def draw_graph(nodes, connections, ax=None):
    edges = [
        (
            i,
            j,
            dict(
                weight=connections[i][j],
                label=(
                    str(round(connections[i][j], 3))
                    if np.abs(connections[i][j]) >= 0.01
                    else ""
                ),
            ),
        )
        for i in connections.keys()
        for j in connections[i].keys()
        if connections[i][j] != 0
    ]
    edges = sorted(edges)
    nodes = sorted(nodes)
    subsets = {i: i[0:-2] for i in nodes}

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    weights = {(i, j): l for i, j, l in G.edges.data("weight")}  # type: ignore
    labels = {(i, j): l for i, j, l in G.edges.data("label")}  # type: ignore
    colors = np.array(list(weights.values()), dtype=np.float64)
    norm = TwoSlopeNorm(vcenter=-0.0)
    colors = norm(colors)

    for n in G.nodes:
        if n[0] not in ["c", "l"]:
            G.nodes[n]["subset"] = "pred"
        else:
            G.nodes[n]["subset"] = subsets[n]
    pos = nx.multipartite_layout(G, subset_key="subset")
    if ax is None:
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(111, frame_on=False)
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        node_size=1000,
        linewidths=0,
        width=5,
        node_color="#bbb",
        node_shape="s",
        arrowstyle="->",
        arrowsize=20,
        edge_cmap=mpl.cm.coolwarm,  # type: ignore
        edge_color=colors,
        connectionstyle="arc,rad=0.1",
        with_labels=False,
    )
    nx.draw_networkx_edge_labels(
        G,
        ax=ax,
        pos=pos,
        edge_labels=labels,
        label_pos=0.35,
        clip_on=False,
        verticalalignment="baseline",
        bbox={"fc": "white", "alpha": 0.0, "ec": "white"},
    )
    nx.draw_networkx_labels(
        G, ax=ax, pos=pos, font_size=14, bbox={"ec": "#555", "fc": "#bbb", "alpha": 0.5}
    )


def draw_graph_with_images(nodes, connections, images, ax=None):
    edges = [
        (
            i,
            j,
            dict(
                weight=connections[i][j],
                label=(
                    f"{round(connections[i][j] * 100, 2)}%"
                    if np.abs(connections[i][j]) >= 0.15
                    else ""
                ),
            ),
        )
        for i in connections.keys()
        for j in connections[i].keys()
        if np.abs(connections[i][j]) >= 0.15
    ]
    edges = sorted(edges)
    nodes = [
        n_node
        for n_node in nodes
        if (n_node in [e[0] for e in edges] or n_node in [e[1] for e in edges])
    ]
    nodes = sorted(nodes)
    subsets = {i: i[0:-2] for i in nodes}

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    weights = {(i, j): l for i, j, l in G.edges.data("weight")}  # type: ignore
    labels = {(i, j): l for i, j, l in G.edges.data("label")}  # type: ignore
    maxv = max(max(weights.values()), 0.0001)
    minv = min(min(weights.values()), -0.0001)
    norm = TwoSlopeNorm(vmin=minv, vcenter=0.0, vmax=maxv)
    colors = norm(np.array(list(weights.values()), dtype=np.float64))

    for n in G.nodes:
        if n[0] not in ["c", "l"]:
            G.nodes[n]["subset"] = "pred"
        else:
            G.nodes[n]["subset"] = subsets[n]
    pos = nx.multipartite_layout(G, subset_key="subset")
    # if ax is None:
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, frame_on=False)
    # ax.set_aspect('equal')
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        node_size=8000,
        linewidths=0,
        width=5,
        node_color="#bbb",
        node_shape="s",
        arrowstyle="->",
        arrowsize=20,
        edge_cmap=mpl.cm.coolwarm,  # type: ignore
        edge_color=colors,
        edge_vmin=minv,
        edge_vmax=maxv,
        # connectionstyle="arc,rad=0.1",
        with_labels=False,
    )

    """ print(pos)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1) """
    nx.draw_networkx_edge_labels(
        G,
        ax=ax,
        pos=pos,
        edge_labels=labels,
        label_pos=0.59,
        font_size=18,
        clip_on=False,
        verticalalignment="baseline",
        bbox={"fc": "white", "alpha": 0.0, "ec": "white"},
    )
    """ nx.draw_networkx_labels(
        G, ax=ax, pos=pos, font_size=14, bbox={"ec": "#555", "fc": "#bbb", "alpha": 0.5}
    ) """
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform  # type: ignore

    piesize = 0.17  # this is the image size
    p2 = piesize / 2
    for n in G.nodes:
        img, norm = images[n]
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes(
            [xa - p2, ya - p2, piesize, piesize],  # type: ignore
        )
        a.set_aspect("equal")
        a.imshow(img, cmap="bwr", norm=norm)
        if n.endswith("0"):
            a.set_title(f"{' '.join(n.split('_')[:-1])}\n{n.split('_')[-1]}")
        else:
            a.set_title(n.split("_")[-1])
        a.yaxis.set_ticks([])
        a.xaxis.set_ticks([])

    a = plt.axes(
        [0.7, 0.1, 0.2, 0.2],  # type: ignore
    )
    a.set_aspect("equal")
    a.imshow(images["original"][0], cmap="Greys")
    a.set_title(images["original"][1])
    a.yaxis.set_ticks([])
    a.xaxis.set_ticks([])


def data_iterations(path, biascut=-1.0, num_it=4):
    with open(path, "r") as f:
        analysis_data = json.load(f)
        alldata = sorted(analysis_data.values(), key=lambda x: x["bias"])

        # to make visualizations more comparable, sort out the bias steps not
        # present in some of them
        less_items = list(np.round(np.linspace(0, 1, 51), 3))
        biases = [a["bias"] for a in alldata]
        datas = [
            list(
                filter(
                    lambda x: x["num_it"] == n and x["bias"] >= biascut,
                    # and x["bias"] in less_items,  # comment out
                    alldata,
                )
            )
            for n in range(num_it)
        ]
        bis = [a["bias"] for a in datas[0]]
    return datas, bis, biases, alldata


def plot_accuracies(
    path, treshold=90, num_it=6, intervened=False, istop=False, isleft=False
):
    datas, bis, biases, alldata = data_iterations(path, num_it=num_it)
    rcol = mpl.cm.winter(np.linspace(0, 1, 4))  # type: ignore
    ecol = mpl.cm.spring(np.linspace(0, 1, 4))  # type: ignore
    fig = plt.figure(figsize=(8, 6))
    fig.set_facecolor(FACECOL)
    plt.ylim([0, 100])

    def to_arr(key, item):
        return np.array(
            [
                [
                    datas[i][x][key][item] for i in range(num_it)
                ]  # [0,2,4,5,7,8,9,10,11,12,13,15]
                for x in range(len(datas[0]))
            ]
        )

    if intervened:
        all_wm_r = to_arr("all_wm_accuracy", 0)
        all_wm_r_sigma = all_wm_r.std(axis=1) / np.sqrt(num_it)
        all_wm_r = all_wm_r.mean(axis=1)
        plt.plot(
            bis,
            all_wm_r,
            c=rcol[2],
            label="only rectangles with watermark",
            linestyle="dashed",
        )
        plt.fill_between(
            bis,
            all_wm_r + all_wm_r_sigma,
            all_wm_r - all_wm_r_sigma,
            facecolor=rcol[2],
            alpha=0.3,
        )
        no_wm_e = to_arr("no_wm_accuracy", 1)
        no_wm_e_sigma = no_wm_e.std(axis=1) / np.sqrt(num_it)
        no_wm_e = no_wm_e.mean(axis=1)
        plt.plot(
            bis,
            no_wm_e,
            c=ecol[1],
            label="only ellipses without watermark",
            linestyle=(0, (1, 1)),
        )
        plt.fill_between(
            bis,
            no_wm_e + no_wm_e_sigma,
            no_wm_e - no_wm_e_sigma,
            facecolor=ecol[1],
            alpha=0.3,
        )
    X1 = to_arr("train_accuracy", 2)
    mu1 = X1.mean(axis=1)
    if not intervened:
        sigma1 = X1.std(axis=1)
        mins = X1.min(axis=1)
        maxs = X1.max(axis=1)
        plt.fill_between(
            bis, maxs, mins, facecolor="#222", alpha=0.1, label="maximum and minimum"
        )
        plt.fill_between(
            bis,
            mu1 + sigma1,
            mu1 - sigma1,
            facecolor="#000",
            alpha=0.3,
            label="standard deviation",
        )

        plt.plot(
            bis,
            to_arr("train_accuracy", 0).mean(axis=1),
            c=rcol[0],
            label="training accuracy rectangles (class 0)",
            linestyle=(0, (1, 1)),
        )
        plt.plot(
            bis,
            to_arr("train_accuracy", 1).mean(axis=1),
            c=ecol[0],
            label="training accuracy ellipses (class 1)",
            linestyle=(0, (3, 1)),
        )

    plt.plot(
        bis,
        mu1,
        c="#000",
        label="training accuracy all",
    )
    bads = [
        [a["num_it"], a["bias"], a["train_accuracy"]]
        for a in list(filter(lambda x: x["train_accuracy"][2] < treshold, alldata))
    ]
    # plt.title("Accuracy of models when intervening on watermark")
    plt.legend(
        bbox_to_anchor=(0.0, 0.01, 1.0, 0.102),
        loc="lower left",
        # ncols=2,
        # mode="expand",
        # borderaxespad=0.0,
        reverse=True,
    )
    if isleft:
        plt.ylabel("Accuracy in \\%")
    else:
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
    if not istop:
        plt.xlabel("Coupling Ratio Rho")
    else:
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)

    return bads


def plot_fancy_distribution(dataset=None, s=[], w=[], bias=0.75, strength=0.5):
    lim_x = [0, 1]  # [np.min(s), np.max(s)]
    lim_y = [0, 1]  # [np.min(w), np.max(w)]

    fig = plt.figure(figsize=(7, 5))
    fig.set_facecolor(FACECOL)
    fig.set_alpha(0.0)
    ax = fig.add_subplot(111)
    ax.set_facecolor(FACECOL)
    ax.set_alpha(0.0)
    TOTAL = 1000
    if dataset is None:

        dataset = DSpritesDataset(bias, strength, False)

        generator = dataset.rng.uniform(0, 1, TOTAL)
        s = dataset.bias * generator + (1 - dataset.bias) * dataset.rng.uniform(
            0, 1, TOTAL
        )
        w = dataset.bias * generator + (1 - dataset.bias) * dataset.rng.uniform(
            0, 1, TOTAL
        )
    plt.scatter(s[:TOTAL], w[:TOTAL], color="#C8D672", s=16)
    plt.ylabel(
        "spurious feature",
    )
    plt.xlabel("shape")

    # ax.tick_params(axis="y", direction="in", pad=-24)
    # ax.tick_params(axis="x", direction="in", pad=-17)
    # ax.yaxis.set_ticks(list(np.round(np.linspace(lim_y[0], lim_y[1], 5), decimals=2)))
    # ax.xaxis.set_ticks(list(np.arange(lim_x[0], lim_x[1], 0.2)))
    plt.text(
        lim_x[0] + 0.01,
        lim_y[1] * 0.65 - dataset.strength,
        "rectangle\nW = 0",
        size=12,
    )
    plt.text(
        lim_x[0] + 0.01,
        lim_y[0] + 0.35 + dataset.strength,
        "rectangle\nW = 1",
        size=12,
    )
    plt.text(lim_x[0] + 0.1, dataset.strength + 0.02, "strength", size=12)
    plt.text(
        lim_x[1] - 0.4,
        lim_y[1] * 0.65 - dataset.strength,
        "ellipse \nW = 0",
        size=12,
    )
    plt.text(
        lim_x[1] - 0.4,
        lim_y[0] + 0.35 + dataset.strength,
        "ellipse\nW = 1",
        size=12,
    )
    plt.title(
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
        (lim_x[0] + lim_x[1]) * 0.5 - 0.1,
        (lim_y[0] + lim_y[1]) * 0.5 - 0.1,
        "1 - $\\rho$",
        c="red",
        size=12,
        fontweight="bold",
        bbox={"fc": "#C8D672", "alpha": 0.8, "ec": "#C8D672"},
    )
    plt.plot(
        [
            (lim_x[0] + lim_x[1]) * 0.5 - (1 - dataset.bias) * 0.15,
            (lim_x[0] + lim_x[1]) * 0.5 + (1 - dataset.bias) * 0.15,
        ],
        [
            (lim_y[0] + lim_y[1]) * 0.5 + (1 - dataset.bias) * 0.15,
            (lim_y[0] + lim_y[1]) * 0.5 - (1 - dataset.bias) * 0.15,
        ],
        c="red",
        linewidth=5,
        alpha=0.8,
    )



def my_plot_grid(images, rows, cols, resize=1, norm=False, cmap="Greys", titles=None):
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
    maxv = max(float(images.abs().max()), 0.001)
    center = 0.0
    divnorm = mpl.colors.TwoSlopeNorm(vmin=-maxv, vcenter=center, vmax=maxv)
    if min(rows, cols) == 1:
        for n in range(max(cols, rows)):
            axs[n].xaxis.set_visible(False)
            axs[n].yaxis.set_visible(False)
            if torch.any(images[n] != 0):
                if not norm:
                    maxv = max(float(images[n].abs().max()), 0.001)
                    # minv = min(float(images[il, n].min()), -0.001)
                    center = 0.0
                    divnorm = mpl.colors.TwoSlopeNorm(
                        vmin=-maxv, vcenter=center, vmax=maxv
                    )
                axs[n].imshow(images[n], cmap=cmap, norm=divnorm)

            else:
                axs[n].imshow(torch.zeros(64, 64), cmap="bwr", norm=divnorm)
                axs[n].text(0.4, 0.5, "is zero")
            if titles is not None:
                axs[n].set_title(titles[n])
    else:
        for il in range(max(rows, 2)):
            if norm == "rows":
                maxv = max(float(images[il].abs().max()), 0.001)
                # minv = min(float(images[il, n].min()), -0.001)
                center = 0.0
                divnorm = mpl.colors.TwoSlopeNorm(vmin=-maxv, vcenter=center, vmax=maxv)
            for n in range(max(cols, 2)):
                axs[il, n].xaxis.set_visible(False)
                axs[il, n].yaxis.set_visible(False)
                if il < rows and n < cols and torch.any(images[il, n] != 0):
                    if not norm:
                        maxv = max(float(images[il, n].abs().max()), 0.001)
                        # minv = min(float(images[il, n].min()), -0.001)
                        center = 0.0
                        divnorm = mpl.colors.TwoSlopeNorm(
                            vmin=-maxv, vcenter=center, vmax=maxv
                        )
                    axs[il, n].imshow(images[il, n], cmap=cmap, norm=divnorm)
                else:
                    axs[il, n].imshow(torch.zeros(64, 64), cmap="bwr", norm=divnorm)
                    axs[il, n].text(0.4, 0.5, "is zero")
                if titles is not None:
                    axs[il, n].set_title(titles[il][n])
                    # axs[il, n].axis("off")
    # return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # Image.fromarray(np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)) #

def plot_dict_grid(
    ref_c: Dict[int, Any],
    cmap_dim=1,
    cmap="bwr",
    vmin=None,
    vmax=None,
    symmetric=True,
    resize: int = None,  # type: ignore
    padding=True,
    figsize=(6, 6),
):
    keys = list(ref_c.keys())
    nrows = len(keys)
    value = next(iter(ref_c.values()))

    if cmap_dim > 1 or cmap_dim < 0 or cmap_dim == None:
        raise ValueError("'cmap_dim' must be 0 or 1 or None.")

    if isinstance(value, Tuple) and isinstance(value[0], Iterable):
        nsubrows = len(value)
        ncols = len(value[0])  # type: ignore
    elif isinstance(value, Iterable):
        nsubrows = 1
        ncols = len(value)
    else:
        raise ValueError(
            "'ref_c' dictionary must contain an iterable of torch.Tensor, np.ndarray or PIL Image or a tuple of thereof."
        )

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(nrows, 1, wspace=0, hspace=0.2)

    for i in range(nrows):
        inner = gridspec.GridSpecFromSubplotSpec(
            nsubrows, ncols, subplot_spec=outer[i], wspace=0, hspace=0.1
        )

        for sr in range(nsubrows):

            if nsubrows > 1:
                img_list = ref_c[keys[i]][sr]
            else:
                img_list = ref_c[keys[i]]

            for c in range(ncols):
                ax = plt.subplot(fig, inner[sr, c])

                if sr == cmap_dim:
                    img = imgify(
                        img_list[c],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        symmetric=symmetric,
                        resize=resize,
                        padding=padding,
                    )
                else:
                    img = imgify(img_list[c], resize=resize, padding=padding)

                ax.imshow(img)  # type: ignore
                ax.set_xticks([])
                ax.set_yticks([])

                if sr == 0 and c == 0:
                    ax.set_ylabel(keys[i])  # type: ignore

                fig.add_subplot(ax)

    outer.tight_layout(fig)
    fig.show()
