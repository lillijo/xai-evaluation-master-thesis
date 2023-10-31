import numpy as np
from expbasics.crp_attribution_binary import CRPAttribution
from expbasics.biased_dsprites_dataset import BiasedDSpritesDataset

SAMPLE_SIZE = 100

def sample_from_categories(ds: BiasedDSpritesDataset):
    indices = {0: {0: [], 1:[]},1: {0: [], 1:[]}}
    for i in range(SAMPLE_SIZE* 20):
        index = np.random.randint(0, len(ds))
        latents, watermark = ds.get_item_info(index)
        if len(indices[latents[0]][int(watermark)]) < SAMPLE_SIZE:
            indices[latents[0]][int(watermark)] += [index]
    return indices   


def average_hierarchies(crpattr: CRPAttribution, indices: dict, sample_size: int = SAMPLE_SIZE):
    relevance_graphs = {
        0: {0: {}, 1: {}},
        1: {0: {}, 1: {}},
        "nodes": set(),
    }
    cluster_data = {}
    for l in indices.keys():
        for w in indices[l].keys():
            for i in range(len(indices[l][w])):
                nodes, edges = crpattr.make_relevance_graph(indices[l][w][i])
                relevance_graphs["nodes"].update(nodes)
                for s in edges.keys():
                    if s not in relevance_graphs[l][w]:
                        relevance_graphs[l][w][s] = {}
                    for t in edges[s].keys():
                        val = edges[s][t] / sample_size
                        if (s, t) not in cluster_data:
                            cluster_data[(s, t)] = [val]
                        else:
                            cluster_data[(s, t)] += [val]
                        if val != 0:
                            if t not in relevance_graphs[l][w][s]:
                                relevance_graphs[l][w][s][t] = val
                            else:
                                relevance_graphs[l][w][s][t] += val
    return relevance_graphs, cluster_data


