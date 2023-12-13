import numpy as np
import torch

from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution

from .crp_attribution import CRPAttribution
from .biased_noisy_dataset import BiasedNoisyDataset
from .test_dataset import get_test_dataset
from .network import load_model
from .ground_truth_measures import GroundTruthMeasures


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


def get_model_etc(bias, num_it=0):
    NAME = "../clustermodels/model"
    model = load_model(NAME, bias, num_it)
    test_ds, test_loader = get_test_dataset()
    gm_dataset = BiasedNoisyDataset()
    gm = GroundTruthMeasures(gm_dataset)
    crp_attribution = CRPAttribution(
        model, test_ds, "model", to_name(bias, num_it)
    )

    return model, gm, crp_attribution, test_ds, test_loader


def get_attributions(
    model, activations, gm, crp_attribution, layer_name="linear_layers.0"
):
    n_samples = 1000
    vlen = len(crp_attribution.layer_id_map[layer_name])
    vector = torch.zeros((n_samples, vlen))
    # activationvector = torch.zeros((n_samples, 6))
    predictions = []
    labels = []
    watermarks = []
    idx = np.round(np.linspace(0, 491519, n_samples)).astype(int)
    for i in range(n_samples):
        img_idx = idx[i]
        img = gm.load_image(img_idx, False)
        # layer_features = model2(img)
        att, predict, label, wm = crp_attribution.relevances2(
            img_idx, activations=activations, layer_name=layer_name
        )
        predictions.append(int(predict))
        labels.append(label)
        watermarks.append(wm)
        vector[i] = att
        # activationvector[i] = layer_features[FEATURE]
    watermarks = np.array(watermarks)
    predictions = np.array(predictions)
    labels = np.array(labels)
    return vector, watermarks, predictions, labels, idx


def get_centroids(dr_res, watermarks, predictions, labels):
    centroids = np.zeros((4, 2))
    for lab in range(2):
        for wm in range(2):
            d = np.logical_and(watermarks == wm, labels == lab)
            centroids[lab + 2 * wm] = np.mean(dr_res[d], axis=0)
    return centroids


def get_attribution_function(model, heatmap=True, batch_size=128, activations=False):
    composite = EpsilonPlusFlat()
    model = model
    cc = ChannelConcept()
    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tdev = torch.device(device)
    attribution = CondAttribution(model, no_param_grad=True, device=tdev)

    def select_max(pred):
        softmax = torch.nn.Softmax(dim=-1)
        id = softmax(pred).argmax(-1).item()
        mask = torch.zeros_like(pred)
        mask[0, id] = pred[0, id]
        return mask

    relu = torch.nn.ReLU()

    def attribution_fn(x):
        x.requires_grad = True
        allatrrs = torch.zeros((batch_size, 6))
        for i, img in enumerate(x):
            attr = attribution(
                img.view(1, 1, 64, 64),
                [{"y": [1]}],
                composite,
                record_layer=layer_names,
                init_rel=select_max,
            )
            if activations:
                rel_c = cc.attribute(
                    relu(attr.activations["linear_layers.0"]), abs_norm=True
                )
            else:
                rel_c = cc.attribute(attr.relevances["linear_layers.0"], abs_norm=True)
            allatrrs[i] = rel_c
        return allatrrs

    def attribution_fn_heatmap(x):
        x.requires_grad = True
        heatmap = torch.zeros((batch_size, 6, 64, 64))
        conditions = [{"y": [1], "linear_layers.0": [i]} for i in range(6)]
        for i, img in enumerate(x):
            for neuron, attr in enumerate(
                attribution.generate(
                    img.view(1, 1, 64, 64),
                    conditions,
                    composite,
                    record_layer=layer_names,
                    batch_size=1,
                    verbose=False,
                )
            ):
                heatmap[i, neuron] = attr.heatmap
        return heatmap

    if heatmap:
        return attribution_fn_heatmap
    return attribution_fn


def get_cavs(model, test_ds, activations=False):
    MAX_INDEX = 491520
    STEP_SIZE = 1000
    single_attr = get_attribution_function(
        model, heatmap=False, batch_size=1, activations=activations
    )

    idx = np.array(list(range(0, MAX_INDEX, STEP_SIZE)))
    cavs = torch.zeros((len(idx), 6))
    for count, index in enumerate(idx):
        x, _ = test_ds[index]
        res = single_attr(x).detach().contiguous()
        cavs[count] = res
    return idx, cavs
