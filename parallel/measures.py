import numpy as np
import torch
from tqdm import tqdm
import json
import copy
from sklearn.metrics import normalized_mutual_info_score

from biased_noisy_dataset import BiasedNoisyDataset
from crp_attribution import CRPAttribution
from network import load_model


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


MAX_INDEX = 491520
SAMPLE_SET_SIZE = 1000
LATSIZE = [2, 2, 6, 40, 32, 32]
SEED = 42
LAYER_ID_MAP = {
    "convolutional_layers.0": 8,
    "convolutional_layers.3": 8,
    "convolutional_layers.6": 8,
    "linear_layers.0": 6,
    "linear_layers.2": 2,
}
NAME = "models/final" #"../clustermodels/model_seeded"  # 
BIASES = list(np.round(np.linspace(0, 1, 51), 3))
ITERATIONS = list(range(16))


class AllMeasures:
    def __init__(self, img_path="", sample_set_size=SAMPLE_SET_SIZE) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.max_index = MAX_INDEX
        self.len_x = sample_set_size
        self.ds = BiasedNoisyDataset(0, 0.5, False, img_path=img_path)
        self.rng = np.random.default_rng()  # seed=SEED

    def diff(self, v1, v2):
        if isinstance(v1, int):
            return abs(v1 - v2)
        return float(torch.sum(torch.abs(v1 - v2))) / v1.shape[0]

    def diff_normed(self, v1, v2):
        return float(
            torch.sum(torch.abs((v1 / (torch.norm(v1))) - (v2 / (torch.norm(v2)))))
        )

    def hm_diff1(self, h0, h1, r0, r1):
        # BIG QUESTIONMARK!!! TODO: how to absolute, euclidean distance etc.
        return float(torch.sum(torch.abs((h1 - h0) * torch.abs(r0[:, None, None]))))

    def hamming_distance(self, h0, h1, w=None):
        if w is not None:
            dist = (
                float(torch.sum(torch.abs((h1 - h0)) * torch.abs(w[:, None, None])))
                / h1.shape[0]
            )
        else:
            dist = float(torch.sum(torch.abs((h1 - h0)))) / h1.shape[0]
        return dist

    def euclidean_distance(self, h0, h1, w=None):
        dist = 0.0
        if torch.any(h0 >= 1) or torch.any(h1 >= 1):
            h0 = h0 / (torch.norm(torch.flatten(h0)) + 1e-10)
            h1 = h1 / (torch.norm(torch.flatten(h1)) + 1e-10)
        if w is not None:
            h0 = h0 * torch.abs(w[:, None, None])
            h1 = h1 * torch.abs(w[:, None, None])
        dist = float(
            torch.sqrt(torch.sum(torch.square(torch.flatten(h1) - torch.flatten(h0))))
        )
        return dist

    def cosine_similarity(self, h0, h1, w=None):
        """
        dist = 0.0
        for c in range(h0.shape[0]):
            if w is not None:
                dist += float(
                    torch.abs(
                        1
                        - torch.nn.functional.cosine_similarity(
                            torch.flatten(h1[c]), torch.flatten(h0[c]), dim=0, eps=1e-10
                        )
                        * w[c]
                    )
                )
            else:
                dist += float(
                    torch.abs(
                        1
                        - torch.nn.functional.cosine_similarity(
                            torch.flatten(h1[c]), torch.flatten(h0[c]), dim=0, eps=1e-10
                        )
                    )
                )
        return dist / h0.shape[0]"""
        if w is not None:
            h0 = h0 * torch.abs(w[:, None, None])
            h1 = h1 * torch.abs(w[:, None, None])
        return float(
            1
            - torch.abs(
                torch.nn.functional.cosine_similarity(
                    torch.flatten(h1),
                    torch.flatten(h0),
                    dim=0,
                )
            )
        )

    def earth_mover_distance(self, h0, h1, w=None):
        pass

    def heatmaps(self, image, layer_name, crpa: CRPAttribution, offset):
        nlen = LAYER_ID_MAP[layer_name]
        conditions = [{layer_name: [i]} for i in range(LAYER_ID_MAP[layer_name])]
        mask = torch.zeros(64, 64).to(self.tdev)
        mask[
            max(0, 57 + offset[0]) : max(0, 58 + offset[0]) + 5,
            max(offset[1] + 3, 0) : max(offset[1] + 4, 0) + 10,
        ] = 1
        mask_size = int(mask.sum())
        heatmaps = torch.zeros((nlen, 64, 64))
        rel_within = []
        rra = []
        rel_total = []
        pointing_game = torch.zeros(nlen)
        non_empty = 0
        for attr in crpa.attribution.generate(
            image,
            conditions,
            crpa.composite,
            start_layer="linear_layers.2",
            record_layer=crpa.layer_names,
            verbose=False,
            batch_size=nlen,
        ):
            heatmaps = attr.heatmap
            heatmaps_abs = attr.heatmap.abs()
            masked = heatmaps_abs * mask
            sorted_values = torch.sort(
                heatmaps_abs.view(nlen, -1), dim=1, descending=True
            ).values
            cutoffs = sorted_values[:, mask_size]
            cutoffs = torch.where(cutoffs > 0, cutoffs, 100)
            rrank = masked >= cutoffs[:, None, None]

            # pointing game
            largest = sorted_values[:, 0]
            non_empty = torch.count_nonzero(largest > 0)
            largest = torch.where(largest > 0, largest, 100)
            pointing_game = (torch.max(masked.view(nlen, -1)) >= largest).float()
            ## rra
            rank_counts = torch.count_nonzero(rrank, dim=(1, 2))
            rra.append(rank_counts / mask_size)

            rel_within.append(torch.sum(masked, dim=(1, 2)))
            rel_total.append(torch.sum(heatmaps_abs, dim=(1, 2)))

        rel_within = torch.cat(rel_within)  # abs_norm* 100
        rel_total = torch.cat(rel_total)
        rra = torch.cat(rra)
        rma = rel_within / (rel_total + 1e-10)
        return dict(
            heatmaps=heatmaps,
            rel_total=rel_total,
            rel_within=rel_within,
            rma=rma,
            rra=rra,
            pg=pointing_game,
            non_empty=non_empty,
        )

    def relevances(self, image, label, layer_name, crpa: CRPAttribution):
        attr = crpa.attribution(
            image,
            [{"y": [label]}],
            crpa.composite,
            record_layer=crpa.layer_names,
        )
        rel_c = crpa.cc.attribute(attr.relevances[layer_name], abs_norm=True)
        return rel_c[0]

    def loop_rho_and_m(self, layer_name):
        models_values = {}
        over_rho = {}
        values = dict(
            m0_mi=0.0,
            m0_rho=0.0,
            m1_mlc=0.0,
            m1_mlc_euclid=0.0,
            # m1_mlc_cosine=0.0,
            m1_pf=0.0,
            m2_mac1=0.0,
            # m2_mac_hamming=0.0,
            # m2_mac_hamming_weigh=0.0,
            m2_mac_euclid=0.0,
            # m2_mac_euclid_weigh=0.0,
            m2_mac_cosine=0.0,
            m2_mac_cosine_2=0.0,
            # m2_mac_weird=0.0,
            m2_rma=0.0,
            m2_rma_val=0.0,
            m2_rma_euclid=0.0,
            m2_rra=0.0,
            m2_rra_euclid=0.0,
            m2_pg=0.0,
            # m2_within=0.0,
            # m2_total=0.0,
        )
        indices = self.rng.choice(MAX_INDEX, self.len_x)
        indices_long = self.rng.choice(MAX_INDEX, 5000)
        for rho in (pbar := tqdm(BIASES)):
            over_rho[rho] = copy.deepcopy(values)
            models_values[rho] = {}

            for m in ITERATIONS:
                pbar.set_postfix(
                    m=m,
                )
                model_name = to_name(rho, m)
                model = load_model(NAME, rho, m)
                crpa = CRPAttribution(model, self.ds, NAME, model_name)
                models_values[rho][m] = copy.deepcopy(values)
                x_values = copy.deepcopy(values)

                for ind, x in enumerate(indices):
                    latents, wm, offset = self.ds.get_item_info(x)
                    mask = torch.zeros(64, 64).to(self.tdev)
                    mask[
                        max(0, 57 + offset[0]) : max(0, 58 + offset[0]) + 5,
                        max(offset[1] + 3, 0) : max(offset[1] + 4, 0) + 10,
                    ] = 1
                    label = latents[0]
                    # image W=0, W=1
                    image_0 = self.ds.load_image_wm(x, False)
                    image_1 = self.ds.load_image_wm(x, True)
                    # prediction output W=0, W=1
                    predv_0 = model(image_0)
                    predv_1 = model(image_1)
                    # classification W=0, W=1
                    predi_0 = int(predv_0.data.max(1)[1][0])
                    predi_1 = int(predv_1.data.max(1)[1][0])
                    # relevances W=0, W=1
                    rel_0 = self.relevances(image_0, label, layer_name, crpa)
                    rel_1 = self.relevances(image_1, label, layer_name, crpa)
                    # heatmaps W=0, W=1
                    hm_0 = self.heatmaps(image_0, layer_name, crpa, offset)
                    hm_1 = self.heatmaps(image_1, layer_name, crpa, offset)

                    x_values["m1_mlc"] += self.diff_normed(predv_1[0], predv_0[0])
                    x_values["m1_mlc_euclid"] += self.euclidean_distance(
                        predv_1[0], predv_0[0]
                    )
                    x_values["m1_pf"] += self.diff(predi_1, predi_0)

                    x_values["m2_mac1"] += self.hm_diff1(
                        hm_1["heatmaps"], hm_0["heatmaps"], rel_1, rel_0
                    )
                    # hamming distance heatmaps, with/without weights
                    """ x_values["m2_mac_hamming"] += self.hamming_distance(
                        hm_1["heatmaps"], hm_0["heatmaps"]
                    )
                    x_values["m2_mac_hamming_weigh"] += self.hamming_distance(
                        hm_1["heatmaps"], hm_0["heatmaps"], rel_1
                    ) """
                    # euclidean distance heatmaps, with/without weights
                    x_values["m2_mac_euclid"] += self.euclidean_distance(
                        hm_1["heatmaps"], hm_0["heatmaps"]
                    )
                    """ x_values["m2_mac_euclid_weigh"] += self.euclidean_distance(
                        hm_1["heatmaps"], hm_0["heatmaps"], rel_1
                    ) """
                    # cosine similarity heatmaps, with/without weights
                    x_values["m2_mac_cosine"] += self.cosine_similarity(
                        hm_1["heatmaps"], hm_0["heatmaps"]
                    )
                    x_values["m2_mac_cosine_weigh"] += self.cosine_similarity(
                        hm_1["heatmaps"], hm_0["heatmaps"], rel_1
                    )
                    if torch.any(hm_1["rma"] != 0):
                        x_values["m2_rma"] += self.diff(hm_1["rma"], hm_0["rma"])
                        x_values["m2_rma_val"] += float(torch.sum(hm_1["rma"]))
                        x_values["m2_rma_euclid"] += self.euclidean_distance(
                            hm_1["rma"], hm_0["rma"]
                        )
                        x_values["m2_rra"] += self.diff(hm_1["rra"], hm_0["rra"])
                        x_values["m2_rra_euclid"] += self.euclidean_distance(
                            hm_1["rra"], hm_0["rra"]
                        )
                        x_values["m2_pg"] += float(
                            torch.sum(torch.abs(hm_1["pg"] - hm_0["pg"]))
                            / hm_1["non_empty"]
                        )

                    # weird self made distance (sth like L1)
                    """ x_values["m2_mac_weird"] += self.diff(
                        hm_1["heatmaps"], hm_0["heatmaps"]
                    ) 
                    x_values["m2_rma_cosine"] += self.cosine_similarity(
                        hm_1["rma"], hm_0["rma"]
                    ) 
                    x_values["m2_within"] += self.diff(
                        hm_1["rel_within"], hm_0["rel_within"]
                    )
                    x_values["m2_total"] += self.diff(
                        hm_1["rel_total"], hm_0["rel_total"]
                    ) """
                for k, v in x_values.items():
                    models_values[rho][m][k] = v / self.len_x
            for m in models_values[rho].keys():
                for k, v in models_values[rho][m].items():
                    over_rho[rho][k] += v / 10
            ds = BiasedNoisyDataset(rho)
            labels_true, labels_pred = [], []
            for ix in indices_long:
                lats, wm, offset = ds.get_item_info(ix)
                labels_true.append(lats[0])
                labels_pred.append(int(wm))
            over_rho[rho]["m0_mi"] = normalized_mutual_info_score(
                labels_true, labels_pred
            )
            over_rho[rho]["m0_rho"] = rho
        with open("over_rho.json", "w") as f:
            json.dump(over_rho, f)
        with open("models_values.json", "w") as f:
            json.dump(models_values, f)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("script_parallel")
    # parser.add_argument("layername", help="layer float", type=str)
    allm = AllMeasures("images/", 100)
    allm.loop_rho_and_m("convolutional_layers.6")  #  #"linear_layers.0"
