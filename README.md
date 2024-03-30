**Master Thesis -- Lilli Joppien**

### Are We Explaining  the Data or the Model?
**Concept-Based Methods and Their Fidelity in Presence of Spurious Features Under a Causal Lense.**

Advisors: Oana-Iuliana Popescu, Simon Bing  
First Examiner: Prof. Dr. Jakob Runge  
Second Examiner: Prof. Dr. Grégoire Montavon  

# Documentation
Next to the conda environment (`env.yml`) some other packages need to be installed to run this experiment

### Installation
1. install conda environment `conda env create -n ENVNAME --file env.yml`
2. follow instructions from https://github.com/rachtibat/zennit-crp to install `zennit-crp`
3. (optionally) install `tigramite` from isntructions at https://github.com/jakobrunge/tigramite/

If one desires to rerun the whole experiment, follow the instructions below,
else create an issue asking for access to the precomputed evaluations

### Computation

1. Training Models
if access to SLURM based cluster with gpu nodes: 
- copy all files inside `experiments` folder to cluster
- make sure that correct `EXPERIMENT` is selected in `run_iterations.py`
- run `script_parallel_iterations.py`
- if accuracies are not satisfactory, rerun until desired accuracy reached
- copy jobs over from cluster and extract into file using `extract_infos.py`
else ask for pretrained models 

2. Computing Explanations and Measures
- run `python3 compute_normal_measures.py [1,2]` (choose 1 for watermark experiment or 2 for pattern)
This computes the relevance and attribution maps measures as well as the region-specific measures (RMA, RRA, PG)
note that if this has not previously been done, it will take about 20 minutes to compute all explanations
- run `python3 compute_relmax.py recompute=True`
- optionally run `python3 compute_latent_factors_gt.py`, this will also take a while when done for the first time.
This computes relevances and model ground truth effect for the other latent factors of our causal model: shape, rotation, scale, posX, posY

3. Explore Measures and Plot Visualizations
- load and analyze cells in `compare_measures.ipynb` 
- for further visualizations also explore `further_visualizations.ipynb`