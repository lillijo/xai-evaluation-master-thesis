from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from PIL import Image
from matplotlib.colors import TwoSlopeNorm

from tigramite import data_processing as pp

# from tigramite.toymodels import structural_causal_processes
from tigramite import plotting as tp


def plot_multipartite(graph_res, layers):
    link_matrix_upper = np.copy(graph_res["graph"])
    link_matrix_upper[:, :, 0] = np.triu(link_matrix_upper[:, :, 0])
    # net = _get_absmax(link_matrix != "")
    net = np.any(link_matrix_upper != "", axis=2)
    G = nx.DiGraph(net)
    node_labels = {}
    for i in G.nodes:
        n = 0
        if i < len(layers[0][1]):
            subset = 0
            layer = layers[0][0]
            n = i
        elif len(layers) > 1 and i < len(layers[0][1]) + len(layers[1][1]):
            subset = 1
            layer = layers[1][0]
            n = i - len(layers[0][1])
        elif len(layers) > 2 and i < len(layers[0][1]) + len(layers[1][1]) + len(
            layers[2][1]
        ):
            subset = 2
            layer = layers[2][0]
            n = i - (len(layers[0][1]) + len(layers[1][1]))
        elif len(layers) > 3 and i < len(layers[0][1]) + len(layers[1][1]) + len(
            layers[2][1]
        ) + len(layers[3][1]):
            subset = 3
            layer = layers[3][0]
            n = i - (len(layers[0][1]) + len(layers[1][1]) + len(layers[2][1]))
        elif len(layers) > 4 and i < len(layers[0][1]) + len(layers[1][1]) + len(
            layers[2][1]
        ) + len(layers[3][1]) + len(layers[4][1]):
            subset = 4
            layer = layers[4][0]
            n = i - (
                len(layers[0][1])
                + len(layers[1][1])
                + len(layers[2][1])
                + len(layers[3][1])
            )
        elif len(layers) > 5 and i < len(layers[0][1]) + len(layers[1][1]) + len(
            layers[2][1]
        ) + len(layers[3][1]) + len(layers[4][1]) + +len(layers[5][1]):
            subset = 5
            layer = layers[5][0]
            n = i - (
                len(layers[0][1])
                + len(layers[1][1])
                + len(layers[2][1])
                + len(layers[3][1])
                + len(layers[4][1])
            )
        elif len(layers) > 6:
            subset = 6
            layer = layers[6][0]
            n = i - (
                len(layers[0][1])
                + len(layers[1][1])
                + len(layers[2][1])
                + len(layers[3][1])
                + len(layers[4][1])
                + len(layers[5][1])
            )
        else:
            subset = 0
            layer = "what"
            n = i

        G.nodes[i]["subset"] = subset
        G.nodes[i]["layer"] = layer
        G.nodes[i]["name"] = layers[subset][1][n]
        node_labels[i] = f"{layer[:3]}{layer[-1]}_{layers[subset][1][n]}"
    pos = nx.multipartite_layout(G, subset_key="subset")
    edge_color = [graph_res["val_matrix"][i][j][0] for (i, j) in G.edges]
    """ node_pos = {"x": [], "y": []}
    for n in pos.keys():
        node_pos["x"] += [pos[n][0]]
        node_pos["y"] += [pos[n][1]]
    tp.plot_graph(
        graph=results["graph"],
        val_matrix=results["val_matrix"],
        save_name=None,
        var_names=var_names,
        figsize=(20, 6),
        arrow_linewidth=4,
        arrowhead_size=30,
        node_size=0.1,
        node_aspect=1,
        label_fontsize=16,
        node_pos=node_pos,
        show_colorbar=False,
    )
    plt.show() """

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, frame_on=False)
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        labels=node_labels,
        node_size=1000,
        linewidths=10,
        width=4,
        node_color="#aaaaaa",
        # arrowstyle="->",
        arrowsize=20,
        edge_cmap=mpl.cm.bwr,  # type: ignore
        edge_color=edge_color,
        connectionstyle="arc3,rad=0.1",
    )


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
                    str(round(connections[i][j] * 100, 2))
                    if np.abs(connections[i][j]) >= 0.03
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
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, frame_on=False)
    # ax.set_aspect('equal')
    nx.draw_networkx(
        G,
        ax=ax,
        pos=pos,
        node_size=8000,
        linewidths=0,
        width=7,
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
        font_size=20,
        clip_on=False,
        verticalalignment="baseline",
        bbox={"fc": "white", "alpha": 0.0, "ec": "white"},
    )
    """ nx.draw_networkx_labels(
        G, ax=ax, pos=pos, font_size=14, bbox={"ec": "#555", "fc": "#bbb", "alpha": 0.5}
    ) """
    trans = ax.transData.transform
    trans2 = fig.transFigure.inverted().transform  # type: ignore

    piesize = 0.18  # this is the image size
    p2 = piesize / 2.0
    for n in G.nodes:
        img, norm = images[n]
        xx, yy = trans(pos[n])  # figure coordinates
        xa, ya = trans2((xx, yy))  # axes coordinates
        a = plt.axes(
            [xa - p2, ya - p2, piesize, piesize],  # type: ignore
        )
        a.set_aspect("equal")
        a.imshow(img, cmap="bwr", norm=norm)
        a.set_title(n)
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
