import numpy as np
import torch
import os
import matplotlib as mpl
from matplotlib import pyplot as plt

from crp.image import vis_opaque_img, plot_grid

# from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, get_output_shapes, abs_norm
from crp.cache import ImageCache
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization
from crp.graph import trace_model_graph
from crp.attribution import AttributionGraph

from expbasics.biased_noisy_dataset import BiasedNoisyDataset


def vis_simple(
    data_batch, heatmaps, rf=False, alpha=1.0, vis_th=0.0, crop_th=0.0, kernel_size=9
):
    return vis_opaque_img(
        data_batch, heatmaps, rf=rf, alpha=0.0, vis_th=0.0, crop_th=0.0
    )


def vis_relevances(
    data_batch, heatmaps, rf=True, alpha=1.0, vis_th=0.0, crop_th=0.0, kernel_size=9
):
    return vis_opaque_img(
        data_batch, heatmaps, rf=rf, alpha=0.1, vis_th=0.0, crop_th=0.0
    )


class CRPAttribution:
    def __init__(self, model, dataset, name, model_name):
        # Feature Visualization:
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # canonizers = [SequentialMergeBatchNorm()]
        self.composite = EpsilonPlusFlat()
        self.dataset: BiasedNoisyDataset = dataset
        self.model = model

        self.cc = ChannelConcept()

        self.layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        self.layer_map = {layer: self.cc for layer in self.layer_names}

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.attribution = CondAttribution(model, no_param_grad=True, device=self.tdev)
        path = f"crp-data/{name}_{model_name}_fv"
        self.fv_path = path
        self.cache = ImageCache(path=path + "-cache")
        self.fv = FeatureVisualization(
            self.attribution, dataset, self.layer_map, path=self.fv_path, cache=self.cache  # type: ignore
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
        print("computing feature vis")
        saved_files = self.fv.run(self.composite, 0, len(self.dataset), 128, 500)
        self.fv.precompute_ref(
            self.layer_id_map,
            plot_list=[vis_simple],
            mode="relevance",
            r_range=(0, 5),
            composite=self.composite,
            batch_size=128,
            stats=True,
        )
        return saved_files

    def make_all_relevances(self, cond_layer, neurons):
        no_ref_samples = 8
        all_refs = {}
        for i in neurons:
            targets, rel = self.fv.compute_stats(
                i, cond_layer, "relevance", top_N=1, norm=True
            )
            ref_c = self.fv.get_stats_reference(
                i,
                cond_layer,
                [targets],
                "relevance",
                (0, no_ref_samples),
                composite=self.composite,
                # rf=True,
                # plot_fn=vis_simple,
            )
            all_refs[f"{i}:{targets}"] = ref_c[f"{i}:{targets}"]
        plot_grid(
            all_refs,
            figsize=(no_ref_samples, len(neurons)),
            padding=True,
            symmetric=True,
        )

    def all_layers_rel(self, image):
        image.requires_grad = True
        """ all_l = sum([len(v) for v in self.layer_id_map.values()])
        relevances = torch.zeros(all_l) """
        relevances = []
        for li, l in enumerate(self.layer_id_map.keys()):
            conditions = [{l: [i]} for i in self.layer_id_map[l]]  # "y": [label],
            attr = self.attribution(
                image,
                [{}],
                self.composite,
                record_layer=self.layer_names,
                start_layer="linear_layers.2",
                init_rel=lambda act: act.clamp(min=0),
            )
            rel_c = self.cc.attribute(attr.relevances[l], abs_norm=True)
            relevances += rel_c.tolist()
        return relevances

    def relevance_for_image(self, label, image, relevances):
        image.requires_grad = True
        lenl = len(self.layer_id_map.keys())
        images = torch.zeros((lenl, 8, 64, 64))
        for li, l in enumerate(self.layer_id_map.keys()):
            conditions = [{l: [i]} for i in self.layer_id_map[l]]  # "y": [label],
            attr = self.attribution(
                image,
                conditions,
                self.composite,
                record_layer=self.layer_names,
                start_layer=l,
                # init_rel = lambda act:  act.clamp(min=0),
            )
            for h in range(attr.heatmap.shape[0]):
                images[li, h] = attr.heatmap[h]
        fig, axs = plt.subplots(
            lenl, 8, figsize=(10, 8), gridspec_kw={"wspace": 0.1, "hspace": 0}
        )
        fig.suptitle("Conditional Heatmap per concept in layer")
        fig.set_facecolor("#2BC4D9")
        fig.set_alpha(0.0)
        for il, l in enumerate(self.layer_id_map.keys()):
            for n in range(8):
                axs[il, n].xaxis.set_visible(False)
                axs[il, n].yaxis.set_visible(False)
                if n < len(self.layer_id_map[l]):
                    axs[il, n].set_title(
                        f"n:{n} {str(relevances[il][n])}%", fontsize=10
                    )  # ,
                    maxv = max(float(images[il, n].max()), 0.001)
                    minv = min(float(images[il, n].min()), -0.001)
                    center = 0.0
                    divnorm = mpl.colors.TwoSlopeNorm(
                        vmin=minv, vcenter=center, vmax=maxv
                    )
                    axs[il, n].imshow(images[il, n], cmap="bwr", norm=divnorm)
                else:
                    axs[il, n].axis("off")
            axs[il, 0].yaxis.set_visible(True)
            axs[il, 0].set_yticks([])
            axs[il, 0].set_ylabel(f"{l[:4]}_{l[-1]}")
        image.requires_grad = False
        axs[lenl - 1, 5].axis("on")
        lab = ["rectangle", "ellipse"]
        axs[lenl - 1, 5].set_title(f"original, predicted: {lab[label]} ({label})")
        axs[lenl - 1, 5].imshow(image[0, 0], cmap="bwr")
        return fig

    def image_info(self, index=None, verbose=False):
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        datum = self.dataset[index]
        label = datum[1]
        img = datum[0]
        latents, watermark, offset = self.dataset.get_item_info(index)
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        result_string = ""
        output = self.model(sample)
        pred = output.data.max(1, keepdim=True)[1]
        res = pred[0][0].tolist()
        conditions = [{"y": [res]}]
        attr = self.attribution(
            sample, conditions, self.composite, record_layer=self.layer_names
        )
        relevances = []
        for cond_layer in self.layer_id_map.keys():
            rel_c = self.cc.attribute(attr.relevances[cond_layer], abs_norm=True)
            act_c = self.cc.attribute(attr.activations[cond_layer], abs_norm=True)
            # concepts ordered by relevance and their contribution to final classification in percent
            rel_values, concept_ids = torch.topk(
                rel_c[0], len(self.layer_id_map[cond_layer])
            )
            relevances += [
                [
                    round(float(rel_c[0][i] * 100), 1)
                    for i in range(len(self.layer_id_map[cond_layer]))
                ]
            ]
            perc = [
                str(int(concept_ids[i]))
                + ": "
                + str(round(float(rel_values[i]) * 100, 2))
                + "%"
                + " act:"
                + str(round(float(act_c[0][int(concept_ids[i])]), 2))
                for i in range(len(self.layer_id_map[cond_layer]))
            ]
            result_string += f'\n \n {cond_layer}: \n {", ".join(perc)} '
        if verbose:
            print(
                f"output: {output.data}, \n latents: {latents}, \n watermark: {watermark}, \n prediction:{res}  {result_string}"
            )
            self.relevance_for_image(res, sample, relevances)
        return relevances

    def relevances(self, index=None, activations=False):
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
            activs = relu(attr.activations["linear_layers.0"])
            relevances = activs
        else:
            relevances = self.cc.attribute(
                attr.relevances["linear_layers.0"][0], abs_norm=True
            )
            # attr.relevances["linear_layers.0"][0]
        return relevances, pred, datum[1], watermark

    def relevances2(self, index=None, activations=False, layer_name="linear_layers.0"):
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

    def get_reference_scores(self, img, label, layer, neurons):
        conditions = [{"y": [label]}]
        attr = self.attribution(
            img, [{}], self.composite, record_layer=self.layer_names, start_layer="linear_layers.2"
        )
        rel_c = self.cc.attribute(attr.relevances[layer], abs_norm=True)  #  activations
        return [rel_c[0][i] for i in neurons]

    def attribute_images(self, imgs, layer_name):
        imgs.requires_grad = True
        attr = self.attribution(
            imgs,
            [{}],
            self.composite,
            record_layer=self.layer_names,
            start_layer=layer_name,
            init_rel=lambda act: act.clamp(min=0),
        )
        return attr.relevances[layer_name]

    def make_relevance_graph(self, index):
        names = {
            "linear_layers.2_0": "0_rectangle",
            "linear_layers.2_1": "1_ellipse",
            # "linear_layers.2_2": "2_heart",
        }
        img, _ = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])

        graph = trace_model_graph(self.model, sample, self.layer_names)
        attgraph = AttributionGraph(self.attribution, graph, self.layer_map)  # type: ignore
        nodes, connections = attgraph(
            sample,
            self.composite,
            pred,
            "linear_layers.2",
            width=[6, 6, 6],
            abs_norm=True,
            verbose=False,
            batch_size=1,
        )
        edges = {}
        for i in connections.keys():
            name = (
                names[f"{i[0]}_{i[1]}"]
                if f"{i[0]}_{i[1]}" in names
                else f"{i[0]}_{i[1]}"
            )
            edges[name] = {f"{j[0]}_{j[1]}": j[2] for j in connections[i]}

        node_labels = [
            names[f"{i[0]}_{i[1]}"] if f"{i[0]}_{i[1]}" in names else f"{i[0]}_{i[1]}"
            for i in nodes
        ]
        return node_labels, edges

    def all_heatmaps(self, image, label, target):
        image.requires_grad = True
        images = {}
        for li, l in enumerate(self.layer_id_map.keys()):
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
                maxv = max(float(heatmap.max()), 0.0001)
                minv = min(float(heatmap.min()), -0.0001)
                center = 0.0
                divnorm = mpl.colors.TwoSlopeNorm(vmin=minv, vcenter=center, vmax=maxv)
                images[f"{l}_{h}"] = [heatmap, divnorm]
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
            width=[6, 2, 2, 2],
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
                if j[2] != 0 and (i[0] == "linear_layers.2" or name in used_nodes):
                    used_nodes.add(name)
                    j_name = f"{j[0]}_{j[1]}"
                    used_nodes.add(j_name)
                    edges[name][j_name] = j[2]
        for source in edges.keys():
            for target in edges[source].keys():
                if in_counts[source[:-2]] > 0:
                    edges[source][target] = (
                        edges[source][target] / in_counts[source[:-2]]
                    )

        node_labels = list(used_nodes)
        return node_labels, edges, images

    def watermark_importance(self, index):
        img, label = self.dataset[index]
        _, watermark, offset = self.dataset.get_item_info(index)
        mask = torch.zeros(64, 64)
        mask[
            max(0, 57 + offset[0]) : max(0, 57 + offset[0]) + 6,
            3 + max(offset[1], 0) : 3 + max(offset[1], 0) + 13,
        ] = 1
        antimask = (mask + 1) % 2
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True

        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])
        conditions = [
            {"y": [pred], l: [i]}
            for l in self.layer_id_map.keys()
            for i in self.layer_id_map[l]
        ]
        relevances = []
        antirelevances = []
        for attr in self.attribution.generate(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,
            exclude_parallel=False,
            batch_size=len(conditions),
            verbose=False,
        ):
            masked = attr.heatmap * mask
            antimasked = attr.heatmap * antimask
            relevances.append(torch.sum(masked, dim=(1, 2)))
            antirelevances.append(torch.sum(antimasked, dim=(1, 2)))

        relevances = torch.cat(relevances)  # for NMF: .clamp(min=0)
        antirelevances = torch.cat(antirelevances) 
        relevances = torch.cat([relevances, antirelevances]) 
        relevances = abs_norm(relevances)

        return dict(
            relevances=relevances,
            watermark=watermark,
            pred=pred,
            label=label,
            output=output.data,
            mask=mask,
        )

    def old_wm_importance(self, index):
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
        masked = attr.heatmap * self.mask[None, :, :]
        antimasked = attr.heatmap * self.antimask[None, :, :]
        sum_watermark_relevance = float(torch.sum(masked, dim=(1, 2)))
        sum_rest_relevance = float(torch.sum(antimasked, dim=(1, 2)))
        return (
            sum_watermark_relevance,
            sum_rest_relevance,
            attr.heatmap,
            output.data,
            label,
        )

    def watermark_neuron_importance(self, index, layer, neuron):
        img, label = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True

        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])

        conditions = [{"y": [pred], layer: [neuron]}]  # pred label

        attr = self.attribution(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,
        )
        masked = attr.heatmap * self.mask[None, :, :]
        antimasked = attr.heatmap * self.antimask[None, :, :]
        sum_watermark_relevance = float(torch.sum(masked, dim=(1, 2)))
        sum_rest_relevance = float(torch.sum(antimasked, dim=(1, 2)))
        return (
            sum_watermark_relevance,
            sum_rest_relevance,
            pred,  # output.data,
            label,
        )

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

    def old_cav_heatmap(self, index, layer, cav):
        img, label = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        output = self.model(sample)
        pred = int(output.data.max(1)[1][0])
        heatmap = torch.zeros(sample.shape)
        conditions = [
            {"y": [pred], layer: [i]} for i in self.layer_id_map[layer]
        ]  # pred label
        for neuron, attr in enumerate(
            self.attribution.generate(
                sample,
                conditions,
                self.composite,
                record_layer=self.layer_names,
                batch_size=1,
                verbose=False,
            )
        ):
            heatmap += attr.heatmap * cav[neuron]
        return abs_norm(heatmap)

    def cav_heatmap(self, index, layer_name, cav):
        self.model.eval()
        img, label = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        self.model(sample)
        cav_s = torch.zeros(1, 8, 7, 7)
        for i in range(cav.shape[0]):
            cav_s[0, i] = cav[i]
        attr = self.attribution(
            sample,
            [{}],
            self.composite,
            start_layer=layer_name,
            init_rel=lambda act: act.clamp(min=0) * cav_s,
        )
        return attr.heatmap

    def watermark_concept_importances(self, indices, test_ds):
        neuron_map = {
            l: {
                int(n): {
                    0: {0: 0.0, 1: 0.0},
                    1: {0: 0.0, 1: 0.0},
                }
                for n in self.layer_id_map[l]
            }
            for l in self.layer_id_map.keys()
        }
        for s in range(2):
            for w in range(2):
                for ind in indices[s][w]:
                    img, label = test_ds[ind]
                    _, wm, _ = test_ds.get_item_info(ind)
                    sample = img.view(1, 1, 64, 64)
                    sample.requires_grad = True
                    conditions = [{"y": [w]}]
                    attr = self.attribution(
                        sample,
                        conditions,
                        self.composite,
                        record_layer=self.layer_names,
                    )
                    for cond_layer in self.layer_id_map.keys():
                        if cond_layer == "linear_layers.2":
                            rel_c = attr.prediction.detach().numpy()
                        else:
                            rel_c = self.cc.attribute(
                                attr.relevances[cond_layer], abs_norm=True
                            )
                        for n in self.layer_id_map[cond_layer]:
                            neuron_map[cond_layer][n][s][w] += float(rel_c[0][n])
        return neuron_map

    def watermark_mask_importance(self, indices, test_ds):
        conditions_w = [
            {l: [n], "y": [1]}
            for l in self.layer_id_map.keys()
            for n in self.layer_id_map[l]
        ]
        conditions_0 = [
            {l: [n], "y": [0]}
            for l in self.layer_id_map.keys()
            for n in self.layer_id_map[l]
        ]
        all_layers = [
            f"{l}_{i}" for l in self.layer_id_map.keys() for i in self.layer_id_map[l]
        ]
        blubtype = {}
        masked_importance = {
            0: {0: blubtype, 1: blubtype},
            1: {0: blubtype, 1: blubtype},
        }
        for s in range(2):
            for w in range(2):
                rels_masked = torch.zeros(len(conditions_0))
                rels_antimasked = torch.zeros(len(conditions_0))
                for ind in indices[s][w]:
                    img, label = test_ds[ind]
                    _, wm, _ = test_ds.get_item_info(ind)
                    sample = img.view(1, 1, 64, 64)
                    assert wm == w and label == s
                    if w == 1:
                        conditions = conditions_w
                    else:
                        conditions = conditions_0
                    rel_c_masked = torch.zeros(len(conditions))
                    rel_c_antimasked = torch.zeros(len(conditions))
                    i = 0
                    if torch.cuda.is_available():
                        sample = sample.cuda()
                    sample.requires_grad = True
                    for attr in self.attribution.generate(
                        sample,
                        conditions,
                        self.composite,
                        record_layer=self.layer_names,
                        batch_size=1,
                        exclude_parallel=False,
                        verbose=False,
                        on_device=self.device,
                    ):
                        masked = attr.heatmap * self.mask[None, :, :]
                        antimasked = attr.heatmap * self.antimask[None, :, :]
                        rel_c_masked[i] = torch.sum(masked, dim=(1, 2))
                        rel_c_antimasked[i] = torch.sum(antimasked, dim=(1, 2))
                        i += 1
                    rels_masked += rel_c_masked
                    rels_antimasked += rel_c_antimasked
                masked_importance[s][w] = {
                    all_layers[k]: {
                        "wm": float(rels_masked[k]),
                        "rest": float(rels_antimasked[k]),
                    }
                    for k in range(len(all_layers))
                }
        return masked_importance
