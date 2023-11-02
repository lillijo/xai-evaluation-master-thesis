import numpy as np
import torch
import os

from crp.image import imgify, vis_opaque_img, plot_grid

# from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, get_output_shapes
from crp.cache import ImageCache
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization
from crp.graph import trace_model_graph
from crp.attribution import AttributionGraph
from crp.helper import abs_norm

from .biased_dsprites_dataset import BiasedDSpritesDataset


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
    def __init__(self, model, dataset, name, strength, bias):
        # Feature Visualization:
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # canonizers = [SequentialMergeBatchNorm()]
        self.composite = EpsilonPlusFlat()
        self.dataset: BiasedDSpritesDataset = dataset
        self.model = model

        self.cc = ChannelConcept()

        self.layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        self.layer_map = {layer: self.cc for layer in self.layer_names}

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.attribution = CondAttribution(model, no_param_grad=True, device=self.tdev)
        path = f'crp-data/{name}_{str(bias).replace("0.", "")}_{str(strength).replace("0.", "")}_fv'
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
        no_ref_samples = 5
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
                rf=False,
                plot_fn=vis_simple,
            )
            all_refs[f"{i}:{targets}"] = ref_c[f"{i}:{targets}"]
        plot_grid(
            all_refs,
            figsize=(no_ref_samples, len(neurons)),
            padding=True,
            symmetric=True,
        )

    def relevance_for_image(self, label, image):
        all_refs = {}
        all_refs["sample"] = torch.ones((6, 64, 64))
        all_refs["sample"][:] = image
        vmin = -1
        vmax = 1
        for l in self.layer_id_map.keys():
            conditions = [{"y": [label], l: [i]} for i in self.layer_id_map[l]]
            attr = self.attribution(
                image, conditions, self.composite, record_layer=self.layer_names
            )
            # vmin = min(vmin, attr.heatmap.min())
            # vmax = max(vmax, attr.heatmap.max())
            all_refs[f"{l[:4]}_{l[-1]}"] = torch.zeros((6, 64, 64))
            for h in range(attr.heatmap.shape[0]):
                all_refs[f"{l[:4]}_{l[-1]}"][h] = attr.heatmap[h]
        plot_grid(
            all_refs,
            figsize=(6, 6),
            padding=False,
            vmax=vmax,
            vmin=vmin,
            symmetric=True,
        )

    def image_info(self, index=None):
        if index is None:
            index = np.random.randint(0, len(self.dataset))
        datum = self.dataset[index]
        label = datum[1]
        img = datum[0]
        latents, watermark = self.dataset.get_item_info(index)
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        result_string = ""
        output = self.model(sample)
        res = int(output.data.round()[0][0].tolist())
        conditions = [{"y": [0]}]
        attr = self.attribution(
            sample, conditions, self.composite, record_layer=self.layer_names
        )
        for cond_layer in self.layer_id_map.keys():
            rel_c = self.cc.attribute(attr.relevances[cond_layer], abs_norm=True)
            # concepts ordered by relevance and their contribution to final classification in percent
            rel_values, concept_ids = torch.topk(
                rel_c[0], len(self.layer_id_map[cond_layer])
            )
            perc = [
                str(int(concept_ids[i]))
                + ": "
                + str(round(float(rel_values[i]) * 100, 2))
                + "%"
                for i in range(len(self.layer_id_map[cond_layer]))
            ]
            result_string += f'\n \n {cond_layer}: \n {", ".join(perc)} '
        print(
            f"output: {output.data[0][0]}, \n latents: {latents}, \n watermark: {watermark}, \n prediction:{res}  {result_string}"
        )
        self.relevance_for_image(0, sample)

    def make_relevance_graph(self, index):
        names = {
            "linear_layers.2_0": "0_rectangle",
            "linear_layers.2_1": "1_ellipse",
            # "linear_layers.2_2": "2_heart",
        }
        img, _ = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True
        graph = trace_model_graph(self.model, sample, self.layer_names)
        attgraph = AttributionGraph(self.attribution, graph, self.layer_map)  # type: ignore
        nodes, connections = attgraph(
            sample,
            self.composite,
            0,
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

    def watermark_importance(self, index):
        img, label = self.dataset[index]
        sample = img.view(1, 1, 64, 64)
        sample.requires_grad = True

        output = self.model(sample)

        conditions = [{"y": [0]}]  # pred label

        attr = self.attribution(
            sample,
            conditions,
            self.composite,
            record_layer=self.layer_names,
            init_rel=output.data[0][0],
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

    def watermark_mask_importance(self, indices, test_ds):
        conditions = [
            {"linear_layers.0": [n], "y": [0]}
            for n in self.layer_id_map["linear_layers.0"]
        ]
        all_layers = [
            f"linear_layers.0_{i}" for i in self.layer_id_map["linear_layers.0"]
        ]
        blubtype = {}
        masked_importance = {
            0: {0: blubtype, 1: blubtype},
            1: {0: blubtype, 1: blubtype},
        }
        for s in range(2):
            for w in range(2):
                rels_masked = torch.zeros(len(conditions))
                rels_antimasked = torch.zeros(len(conditions))
                for ind in indices[s][w]:
                    img, label = test_ds[ind]
                    _, wm = test_ds.get_item_info(ind)
                    sample = img.view(1, 1, 64, 64)
                    assert wm == w and label == s
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
