from shape_covnet import train_network
from causal_dsprites_dataset import CausalDSpritesDataset
from torch.utils.data import DataLoader

import numpy as np
import torch

from crp.image import vis_opaque_img
from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, get_output_shapes
from crp.cache import ImageCache
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization

BATCHSIZE = 128


def main():
    dsprites_dataset_train = CausalDSpritesDataset(
        train=True, verbose=True, with_watermark=True
    )
    dsprites_dataset_test_unbiased = CausalDSpritesDataset(
        train=False, with_watermark=True, causal=False
    )
    training_loader = DataLoader(
        dsprites_dataset_train, batch_size=BATCHSIZE, shuffle=True
    )

    model = train_network(training_loader, BATCHSIZE, False)
    composite = EpsilonPlusFlat()

    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(model, no_param_grad=True)

    def vis_simple(
        data_batch,
        heatmaps,
        rf=False,
        alpha=1.0,
        vis_th=0.0,
        crop_th=0.0,
        kernel_size=9,
    ):
        return vis_opaque_img(
            data_batch, heatmaps, rf=rf, alpha=0.0, vis_th=0.0, crop_th=0.0
        )

    fv_path = "crp-stuff/dsprites-causal-unbiased-ds"
    cache = ImageCache(path="causal-unbiased-cache")

    fv = FeatureVisualization(attribution, dsprites_dataset_test_unbiased, layer_map, path=fv_path, cache=cache)  # type: ignore
    # RUN REFERENCE IMAGE COMPUTATION
    saved_files = fv.run(composite, 0, len(dsprites_dataset_test_unbiased), 20, 100)

    output_shape = get_output_shapes(model, fv.get_data_sample(0)[0], layer_names)
    layer_id_map = {
        l_name: np.arange(0, out[0]) for l_name, out in output_shape.items()
    }

    fv.precompute_ref(
        layer_id_map,
        plot_list=[vis_simple],
        mode="relevance",
        r_range=(0, 10),
        composite=composite,
        batch_size=32,
        stats=True,
    )
    return model, fv


if __name__ == "__main__":
    main()
