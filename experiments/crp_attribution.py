import numpy as np
import torch
from os.path import isfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from torchvision.transforms.functional import gaussian_blur

from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from crp.image import vis_opaque_img, plot_grid, imgify, get_crop_range
from crp.concepts import ChannelConcept, Concept
from crp.helper import get_layer_names, get_output_shapes, abs_norm, max_norm
from crp.cache import ImageCache
from crp.attribution import CondAttribution, AttributionGraph
from crp.visualization import FeatureVisualization
from crp.graph import trace_model_graph

from wdsprites_dataset import DSpritesDataset
from test_dataset import TestDataset
from plotting import plot_dict_grid


def vis_simple(
    data_batch,
    heatmaps,
    rf=False,
    alpha=1.0,
    vis_th=0.1,
    crop_th=0.005,
    kernel_size=21,
    cmap="bwr",
    vmin=None,
    vmax=None,
    symmetric=True,
):
    img_list = []
    for i in range(len(data_batch)):
        img = data_batch[i]
        heat = heatmaps[i]
        if rf:
            filtered_heat = max_norm(
                gaussian_blur(heat.unsqueeze(0), kernel_size=kernel_size)[0]  # type: ignore
            )
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
            img_t = img[..., row1:row2, col1:col2]

            if img_t.sum() != 0:
                # check whether img or vis_mask is not empty
                img = img_t

        maxv = max(float(img.abs().max()), 0.001)
        img = imgify(img, cmap=cmap, vmin=-maxv, vmax=maxv, symmetric=True)

        img_list.append(img)

    return img_list


def vis_img_heat(
    data_batch,
    heatmaps,
    rf=True,
    crop_th=0.1,
    kernel_size=19,
    cmap="bwr",
    vmin=None,
    vmax=None,
    symmetric=True,
):

    img_list, heat_list = [], []

    for i in range(len(data_batch)):

        img = data_batch[i]
        heat = heatmaps[i]

        if rf:
            filtered_heat = max_norm(
                gaussian_blur(heat.unsqueeze(0), kernel_size=kernel_size)[0]  # type: ignore
            )
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
            img_t = img[..., row1:row2, col1:col2]
            heat_t = heat[row1:row2, col1:col2]

            if img_t.sum() != 0 and heat_t.sum() != 0:
                # check whether img or vis_mask is not empty
                img = img_t
                heat = heat_t

        heat = imgify(heat, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric)
        maxv = max(float(img.abs().max()), 0.001)
        img = imgify(img, cmap=cmap, vmin=-maxv, vmax=maxv)

        img_list.append(img)
        heat_list.append(heat)

    return img_list, heat_list


def get_bbox(
    data_batch,
    heatmaps,
    rf=True,
    crop_th=0.1,
    kernel_size=19,
    cmap="bwr",
    vmin=None,
    vmax=None,
    symmetric=True,
):
    # instead of putting out ready images, this function only
    # delivers the receptive field per neuron and reference image
    heat_list = []
    for i in range(len(data_batch)):
        heat = heatmaps[i]
        if rf:
            filtered_heat = max_norm(
                gaussian_blur(heat.unsqueeze(0), kernel_size=kernel_size)[0]  # type: ignore
            )
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
            heat_list.append([row1, row2, col1, col2])

    return heat_list


class CRPAttribution:
    def __init__(self, model, dataset, name, model_name, max_target="sum"):
        # Feature Visualization:
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # canonizers = [SequentialMergeBatchNorm()]
        self.composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
        self.dataset: DSpritesDataset = dataset
        self.model = model

        self.cc: Concept = ChannelConcept()

        self.layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        self.layer_map = {layer: self.cc for layer in self.layer_names}

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.attribution = CondAttribution(model, no_param_grad=True, device=self.tdev)
        path = f"crp-data/{model_name}_fv"
        self.fv_path = path
        self.cache = ImageCache(path=self.fv_path + "-cache")
        self.max_target = max_target
        self.fv = FeatureVisualization(
            self.attribution,
            self.dataset,
            self.layer_map,  # type: ignore
            path=self.fv_path,
            cache=self.cache,  # type: ignore
            max_target=self.max_target,
        )
        self.output_shape = get_output_shapes(
            model, self.fv.get_data_sample(0)[0], self.layer_names
        )
        self.layer_id_map = {
            l_name: np.arange(0, out[0]) for l_name, out in self.output_shape.items()
        }
        mask = torch.zeros(64, 64)
        mask[52:, 0:17] = 1
        antimask = torch.ones(64, 64)
        antimask[52:, 0:17] = 0
        if torch.cuda.is_available():
            antimask = antimask.cuda()
            mask = mask.cuda()
        self.mask = mask
        self.antimask = antimask

    def compute_feature_vis(self):
        if not isfile(
            f"{self.fv_path}/ActMax_{self.max_target}_normed/linear_layers.0_rel.npy"
        ):
            print("computing feature vis")
            print("len dataset", len(self.dataset))
            saved_files = self.fv.run(
                self.composite, 0, len(self.dataset), 128, len(self.dataset)
            )
        else:
            print("feature vis is computed")
            saved_files = []
        return saved_files

    def make_stats_references(
        self, cond_layer, neurons, relact="relevance", relevances=[]
    ):
        no_ref_samples = 6
        all_refs = {}
        for i in neurons:
            targets, rel = self.fv.compute_stats(
                i, cond_layer, relact, top_N=2, norm=True
            )
            for t in [1, 0]:
                ref_c = self.fv.get_stats_reference(
                    i,
                    cond_layer,
                    [t],
                    relact,
                    (0, no_ref_samples),
                    composite=self.composite,
                    rf=True,
                    plot_fn=vis_simple,  # vis_img_heat,
                )
                if t == 0:
                    all_refs[f"$c_{i},s_{t}$ {torch.round(relevances[t,i]*100)}\\%"] = (
                        ref_c[f"{i}:{t}"]
                    )
                else:
                    all_refs[f" $s_{t}$ {torch.round(relevances[t,i]*100)}\\%"] = ref_c[
                        f"{i}:{t}"
                    ]
        plot_grid(
            all_refs,
            figsize=(no_ref_samples, 2 * len(neurons)),
            padding=True,
            symmetric=True,
        )

    def make_all_references(self, cond_layer, neurons, relact="relevance"):
        no_ref_samples = 8
        ref_c = self.fv.get_max_reference(
            neurons,
            cond_layer,
            relact,
            (0, no_ref_samples),
            composite=self.composite,
            rf=False,
            plot_fn=vis_img_heat,  #
        )
        plot_dict_grid(
            ref_c,
            figsize=(2 * no_ref_samples, 4 * len(neurons)),
            padding=True,
            symmetric=False,
            cmap="Greys",
            cmap_dim=1,
        )

    def relevance_for_image(self, label, image, relevances, pred):
        image.requires_grad = True
        lenl = len(self.layer_id_map.keys())
        images = torch.zeros((lenl, 8, 64, 64))
        for li, l in enumerate(self.layer_id_map.keys()):
            conditions = [
                {
                    l: [i],
                    "y": [pred],
                }
                for i in self.layer_id_map[l]
            ]  # "y": [label],
            attr = self.attribution(
                image,
                conditions,
                self.composite,
                record_layer=self.layer_names,
                # start_layer=l,
                # init_rel = lambda act:  act.clamp(min=0),
            )
            for h in range(attr.heatmap.shape[0]):
                images[li, h] = attr.heatmap[h]
        fig, axs = plt.subplots(
            lenl, 8, figsize=(20, 16), gridspec_kw={"wspace": 0.1, "hspace": 0.2}
        )
        # fig.suptitle("Concept-Conditional Heatmap per concept in layer")
        fig.set_alpha(0.0)
        for il, l in enumerate(self.layer_id_map.keys()):
            for n in range(8):
                axs[il, n].xaxis.set_visible(False)
                axs[il, n].yaxis.set_visible(False)
                if n < len(self.layer_id_map[l]):
                    axs[il, n].set_title(
                        f"{str(round(relevances[il][n],1))} \\%",
                        fontsize=20,  # {str(round(relevances[il][n],1))}%
                    )  # ,
                    axs[il, n].text(
                        1,
                        1,
                        f"concept {n}",
                        size=14,
                    )
                    maxv = max(float(images[il, n].abs().max()), 0.0001)
                    # minv = min(float(images[il, n].min()), -0.0001)
                    center = 0.0
                    divnorm = mpl.colors.TwoSlopeNorm(
                        vmin=-maxv, vcenter=center, vmax=maxv
                    )
                    axs[il, n].imshow(images[il, n], cmap="bwr", norm=divnorm)
                else:
                    axs[il, n].axis("off")
            axs[il, 0].yaxis.set_visible(True)
            axs[il, 0].set_yticks([])
            axs[il, 0].set_ylabel(l.replace("_layers", ""))  # f"{l[:4]}_{l[-1]}")
        image.requires_grad = False
        axs[lenl - 1, 5].axis("on")
        lab = ["rectangle", "ellipse"]
        axs[lenl - 1, 5].set_title(
            f"original {lab[label]} ({label}), predicted: {lab[pred]} ({pred})"
        )
        maxv = max(float(image[0, 0].abs().max()), 0.0001)
        center = 0.0
        divnorm = mpl.colors.TwoSlopeNorm(vmin=-maxv, vcenter=center, vmax=maxv)
        axs[lenl - 1, 5].imshow(image[0, 0], cmap="coolwarm", norm=divnorm)
        return fig

    def image_info(self, index=None, verbose=False, onlywm=False):
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        datum = self.dataset[index]
        label = datum[1]
        img = datum[0]
        latents, watermark, offset = self.dataset.get_item_info(index)
        if onlywm:
            sample = self.dataset.load_image_wm(index, not watermark)
        else:
            sample = img.view(1, 1, 64, 64)
            sample.requires_grad = True
        result_string = ""
        output = self.model(sample)
        pred = output.data.max(1, keepdim=True)[1]
        res = pred[0][0].tolist()
        conditions = [{"y": [res]}]
        attr = self.attribution(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,  # , init_rel=lambda x: x.clamp(min=0)
        )
        relevances = []
        for cond_layer in self.layer_id_map.keys():
            """conditions = [{cond_layer: self.layer_id_map[cond_layer]}]
            attr = self.attribution(
                sample, conditions, self.composite, record_layer=self.layer_names, init_rel=lambda x: x
            )"""
            rel_c = self.cc.attribute(attr.relevances[cond_layer])  # , abs_norm=True)
            act_c = self.cc.attribute(attr.activations[cond_layer], abs_norm=True)
            # print(attr.relevances[cond_layer].shape, torch.where(attr.relevances[cond_layer] != 0))
            # concepts ordered by relevance and their contribution to final classification in percent
            relevances += [
                [
                    float(rel_c[0][i] * 100)
                    for i in range(len(self.layer_id_map[cond_layer]))
                ]
            ]
            perc = [
                str(i) + ": " + str(round(float(act_c[0][i]), 2))
                for i in range(len(self.layer_id_map[cond_layer]))
            ]
            result_string += f'\n {cond_layer}: \n {", ".join(perc)} '
        if verbose:
            print(
                f"output: {output.data}, \n latents: {latents}, watermark: {watermark}, prediction:{res} {result_string}"
            )
            self.relevance_for_image(label, sample, relevances, res)
        return relevances

    def relevances_of_layer(
        self, index=None, activations=False, layer_name="linear_layers.0"
    ):
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        datum = self.dataset[index]
        img = datum[0]
        _, watermark, _ = self.dataset.get_item_info(index)
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        output = self.model(sample)
        pred = output.data.max(1, keepdim=True)[1]
        res = pred[0][0].tolist()
        conditions = [{"y": [res]}]
        attr = self.attribution(
            sample, conditions, self.composite, record_layer=self.layer_names
        )
        if activations:
            relu = torch.nn.ReLU()
            activs = relu(attr.activations[layer_name])
            relevances = activs
        else:
            relevances = self.cc.attribute(attr.relevances[layer_name], abs_norm=True)
            # attr.relevances[layer_name][0]
        return relevances, pred, datum[1], watermark

    def all_heatmaps(self, image, label, target):
        image.requires_grad = True
        images = {}
        for li, l in enumerate(self.layer_id_map.keys()):
            max_all = 0
            conditions = [{l: [i]} for i in self.layer_id_map[l]]
            attr = self.attribution(
                image,
                conditions,
                self.composite,
                record_layer=self.layer_names,
                start_layer=l,
            )
            for h in range(attr.heatmap.shape[0]):
                heatmap = attr.heatmap[h]
                maxv = max(float(heatmap.abs().max()), 0.0001)
                max_all = max(maxv, max_all)
                # minv = min(float(heatmap.min()), -0.0001)
                center = 0.0
                divnorm = mpl.colors.TwoSlopeNorm(vmin=-maxv, vcenter=center, vmax=maxv)
                images[f"{l}_{h}"] = [heatmap, None]
            center = 0.0
            divnorm = mpl.colors.TwoSlopeNorm(
                vmin=-max_all, vcenter=center, vmax=max_all
            )
            for h in range(attr.heatmap.shape[0]):
                images[f"{l}_{h}"][1] = divnorm

        images["original"] = [
            image[0, 0].detach().numpy(),
            f"prediction: {label} label: {target}",
        ]
        return images

    def complete_relevance_graph(self, index):
        img, target = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])
        images = self.all_heatmaps(sample, pred, target)
        graph = trace_model_graph(self.model, sample, self.layer_names)
        attgraph = AttributionGraph(self.attribution, graph, self.layer_map)  # type: ignore
        nodes, connections = attgraph(
            sample=sample,
            composite=self.composite,
            concept_id=pred,
            layer_name="linear_layers.2",
            width=[6, 8],  # [8, 6],  #
            abs_norm=True,
            verbose=False,
            batch_size=1,
        )
        edges = {}
        used_nodes = set()
        in_counts = {f"{i[0]}": 0 for i in nodes}
        for i in connections.keys():
            name = f"{i[0]}_{i[1]}"
            edges[name] = {}
            in_counts[i[0]] += 1
            for j in connections[i]:
                if np.abs(j[2]) > 0.05 and (
                    i[0] == "linear_layers.2" or name in used_nodes
                ):
                    used_nodes.add(name)
                    j_name = f"{j[0]}_{j[1]}"
                    used_nodes.add(j_name)
                    edges[name][j_name] = j[2]
        for source in edges.keys():
            for target in edges[source].keys():
                if in_counts[source[:-2]] > 0:
                    edges[source][target] = edges[source][
                        target
                    ]  # / in_counts[source[:-2]]
        node_labels = list(used_nodes)
        return node_labels, edges, images

    def heatmap(self, index):
        img, label = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])
        conditions = [{"y": [pred]}]  # pred label
        attr = self.attribution(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,
        )
        return attr.heatmap, pred

    def heatmap_neuron(self, index, neuron, layer):
        if isinstance(index, int):
            img, label = self.dataset[index]
            sample = img.view(1, 1, 64, 64)
            sample.requires_grad = True
        else:
            sample = index
        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])
        conditions = [{"y": [pred], layer: [neuron]}]  # pred label
        attr = self.attribution(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,
        )
        return attr.heatmap, sample, pred

    def heatmap_given_img(self, img, pred):
        sample = img.view(1, 1, 64, 64)
        conditions = [{"y": [pred]}]  # pred label
        attr = self.attribution(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,
        )
        return attr.heatmap
