## Causal Games and Causal Nash Equilibrium

Causal Decision Problem (CDP) using Causal Graphical Model (CGM):

- (\Omega, A, G, C, E) -> \Omega events, A available actions, C consequences, E algebra of events over \Omega, G causal graph
- action a is preferable over action b (a>=b) if sum of probabilities given do(a) >= given do(b):
  sum u(c) ( sum P_g (c| do(a)*P_C(g))) >= sum u(c) ( sum P_g (c| do(b)*P_C(g)))
- Bayesian Causal Games = strategic game with N players, no information about other players actions or thought process (=private information)
- true causal model is unknown but fixed

## Causal Discovery as a Game

- 2-player game: nature against scientist
- how many experiments (intervention on a single variable) need to be performed to find true causal graph
- 2 for 2 variables, N-1 for N variables in the worst case
- much harder if experiments intervening on > 1 variable possible, but theoretic worst case for this is not much better
- possible directions:
  - additional assumptions like reduced information set
  - cost for e.g. sample size, ethical cost
  - constraints/background knowledge on causal structure, or constraints on possible experiments
  - robustness of search strategy if nature is not optimal adversary

## A spatiotemporal stochastic climate model for benchmarking causal discovery methods for teleconnections

- based either on time series at grid locations or modes of variability time series through dimension reduction
- highly interdependent spatiotemporal dynamical system
- related: "climate network analysis"
- teleconnection analysis approach
- 2 dimensions: correlation-causation, gridlevel-modelevel
- While our experiments showed that Varimax works better than PCA, we do not claim that Varimax is a
  valid estimator of the SAVAR model weights. Since the mode definition in SAVAR comes from the
  distinction between fast dynamics encoded in Σy in model (16) and time-delayed teleconnections encoded
  in Φ, another dimension-reduction method that takes into account not just the zero-lag covariance matrix,
  but also lagged covariances, might be even better suited

## A million variables and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images

- also finds Markov Blankets of any variable (markov blanket = parents, children, other parents of children)
- finds Markov Equivalence Class of a Bayesian Network
- what is "Pearson correlation", "Markov Random Field", "GLASSO and LASSO", "Markov Blankets", "GES" ?
- structure prior, sample prior
- "one-edge faithfulness" no uncorrelated edge is added -> fails for cancelling effects

## Causal Inference and causal explanation with background knowledge - Meek, Greedy Equivalence Search

- is there causal explanation consistent with background knowledge which explains observed independencies?
- if yes, what is MEC (markov equivalence class) - "causal relationships common to every such explanation"
- is basically PC algorithm just with (unnecessary) extra rule and with skeleton already existing

## Statistically Efficient Greedy Equivalence Search

- making assumptions and only looking for scorings with less than k variables conditioned on
- SE-GES to GES is what PC is to SGS
- hybrid approach that uses both a score and an independence oracle
- starting with complete graph and iteratively inccreasing independece sets cardinality
- only applying delete operator (deleting edge) in certain cases

## Inferring causation from time series in Earth system sciences

- arctic climate / sea ice / methane as a topic? [25]
- challenges in causal inference (for climate time series) -> image
- other methods discussed
- each methods with its limitations/challenges, what could be further researched on

## D’ya like DAGs? A Survey on Structure Learning and Causal Discovery

- very extensive definitions for all concepts (can reuse probably)
- information geometric properties -> non linear noise
- evaluation metrics for causal inference tasks: True Positive Rate (TPR), False Positive Rate (FPR), Area Over Curve (AOC), Structural Hamming Distance (SHD), Structural Interventional Distance (SID)
- very many algorithms and methods, particularily "NO TEARS" variations

## Complex networks reveal global pattern of extreme-rainfall teleconnections (Niklas Boers)

- connections follow "power-law" fit up to ~2500km, afterwards one finds teleconnections
- have to remove "random" correlations and artifacts introduced through high dimensionality
- connections between the Eurasian wave train and the Indian summer monsoon

## Identifying causal gateways and mediators in complex spatio-temporal systems

- susceptibilities to volcanic eruptions, extreme events or **geoengineering**
- complex system’s susceptibility to perturbations
- approach: reduce gridded dataset using Varimax, causal discovery, causal effect quantification

## Using Causal Effect Networks to Analyze Different Arctic Drivers of Midlatitude Winter Circulation

- time series analysis causal effect networks (CEN)
- Severe winters in northern hemisphere midlatitudes -> something about negative arctic oscillation
- application of a simple PCMCI to climate data, trying out different significance alphas and time lags

## Multi-Level Cause-Effect Systems

- find macro-level causal effects from arrays of micro-measurements
- examples micro/macro: climate data/large scale patterns, brain scans/cognition, gene sequencing/health outcomes
- learn "fundamental causal partition" - coarsest desccription retaining all causal information
- prove "causal coarsening theorem"
- under which conditions multi-level causal descriptions

## Unsupervised Discovery of El Niño Using Causal Feature Learning on Microlevel Climate Data

- how well does a partition of T (temperature) separate El Nino and La Nina events
- basically a clustering method

## Causal feature learning: an overview

- "Just about any scientific discipline is concerned with developing ‘macrovariables’ that summarize an underlying finer-scale structure of ‘microvariables’"
- a parallel approach to PCMCI -> not explicitly modelling for time series but just seeing a "cluster" of micro-variables/states as one macro variable
- ambiguous manipulation: "low-cholesterol" -> there is an appropriate level of aggregation

## Causal network reconstruction from time series: From theoretical assumptions to practical estimation

- overview paper, comparing Runges approach to others
- using Conditional Mutual Information (CMI) as independence test
- While classical statistical methods are often based on the assumption of linearity (which allows us to derive rigorous results), modern statistics, the physics community, and the recent field of machine learning have developed non-parametric or model-free methods that allow us to better capture the nonlinear reality of many dynamical complex systems—at the cost of weaker theoretical result
- PCMCI really bad for deterministic relationships
- other Methods tested: FullCI, OCE, and CCM

## CAUSALITY FOR MACHINE LEARNING - Bernhard Schölkopf - 2019

- essay -> really good for motivation
- information revolution
- AI used Pearls bayesian networks etc. but not with causality explicit
- biggest problem: AI only uses IID (independent and identically distributed) data
- principle of (algorithmically) independent mechanisms implys "second law of thermodynamics" = "Arrow of time"
- apply causality to machine learning
- semi-supervised learning can only learn non-causal / anticausal relationships. otherwise residual is 0 in X->Y direction
- causal direction may protect from adversarial attacks (minute changes to input)
- reinforcement learning (RL) - humans identify objects, for machine game hs to be downsampled
- humans use time data -> can identify invariant causal features, machine is better for permutated data
- "One way to do this is to employ model-based RL using SCMs, an approach which can help address a problem of confounding in RL where time-varying and time-invariant unobserved confounders influence both actions and rewards. In such an approach, nonstationarities would be a feature rather than a bug, and agents would actively seek out regions that are different from the known ones in order to challenge their existing model and understand which components are robust."
- "learning causal models that contain independent mechanisms helps in transferring modules across substantially different domains"

## Active Learning of Causal Structures with Deep Reinforcement Learning

- using RL to find optimal experiment design (to quickly cover search space)

## Spatio-temporal Autoencoders in Weather and Climate Research

- encoder + decoder nn -> encode efficiently in bottleneck --- spatio-temporal
- many autoencoders:
  - GAE generalized -> reconstruct neighbourhoods w. knn
  - SAE Sparse -> bottleneck not necessarily smaller but sparser
  - DAE denoising -> data is low dimensional but noisy, use corrupted input for training, to reconstruct clean input
  - CAE contractive -> minimize frobenius norm
  - VAE variational -> map unknown distribution of inputs to d-dimensional multivariate gaussian distribution and then back (not single input)
  1. Maximize likelihood of inputs under output distribution (using "evidence lower bound" ELBO minimize Kullback-Leibler ("Jensen's inequality)
  2. Force latent distribution to multivariate standard normal
     Problem: sampling process has no derivative therefore only derive m+s and apply to standard gaussian sample -> derivative for m + s
     VAE can bethough ot as finding optimal kernel for kernel-PCA
- use of Latent Space of VAE:
  - as dimensionality reduction
  - as feature extractor for prediction
  - BUT: loss of interpretability
- use of Output of VAE:
  - as sample generator
  - as sample denoiser
  - as anomaly detector

## Causal Explanations and XAI

- XAI because we try to make predictions about results of actions and not observations
- need causal explanations define: sufficient explanations and counterfactual explanations
- use feature attribution for explanation of ml model
- **sufficient explanation** under which conditions an action guarantees a particular output
- **counterfactual explanation** finding an action that changes an observed input to produce a change in an already observed output
- **actual causes** which past actions actually explain an already observed output. an actual cause is a part of a good sufficient explanation for which there exist counterfactual values that would not have made the explanation better
- relation between **prediction** and **explanation**: good explanation needs to predict what would happen to explanandum if explanans had been different
- must also state which factors _may not be manipulated_ for explanation to hold
- _weak_ sufficiency: if X=x -> Y=y, _strong_ sufficiency: if X=x and variables in N (including Y) safeguarded from interventions -> Y=y
- this notion is similar to the "forbidden set" for adjustment criteria
- counterfactual explanations:
  - direct counterfactual dependence: all things same: if X=x' then Y=y' --- W = V \ {X U Y} (all vars)
  - standard counterfactual dependence: other variables might change because of X=x' --- W = empty set
  - non-standard non-direct counterfactual dependence: X=x' and witnesses W held fixed => Y=y'
- _actual cause_ part of good sufficient explanation in which number of variables cannot be reduced
- counterfactual fairness: usually output should not standardly depend on protected variable
- should only counterfactually depend along _unfair paths_
- Therefore we can define fairness by demanding that protected variables do not cause the outcome along an unfair network, i.e., a network that consists entirely of unfair paths.

## XAI Methods for Neural Time Series Classification: A Brief Review

- a) predicting a sample given a set of samples, b) assign a sample to one of known groups of samples
- for deep learning algorithms for time-series classification
- MLP (multilayer perceptron) -- not good for sequential data
- RNN (recurrent neural network) -- slow to train
- LSTM (long short-term memory)
- GRU
- CNN (convolutional nn) -- fast training (paralellization), noise resistant, capture cross-channel correlations
- deconvolutional network
- other non-deep-learning methods: COTE, HIVE-COTE, BOSS, InceptionTime...
- **interpretability** the ability to explain or to present in understandable terms to a human
- worse than human: what went wrong, same: build trust, better: learn from it
- CAM class activation maps
- multilevel Wavelet Decomposition Network (mWDN)
- sensitivity analysis: model agnostic, measures the importance of each feature by perturbing it and observing the change in the classifiers output
- all applied to _LSTMs_ and _ResNets_:
- (occlusion based) attention mechanisms
- saliency maps - visualization technique like importance heatmap
- Layer-wise Relevance Propagation (LRP)
- DeepLIFT
- Local Interpretable Model-Agnostic Explanations LIME
- SHapley Additive exPlanations SHAP

## BaCaDI: Bayesian Causal Discovery with Unknown Interventions

- for finding intervention targets (in experimentation)
- is propagating the epistemic uncertainty across all latent quantities
- operates in the continuous space of latent probabilistic representations of both causal Bayesian networks (CBNs) and interventions
- imperfect (soft) intervention: dependence to parents remains vs. perfect (hard, structural)
- In active learning of CBNs, a commonly used f is the expected information gain about G after certain interventions
- uses Sparse Mechanism Shift Hypothesis -> distribution only slightly varies
- algorithm uses conditional independence tests

## Network-based forecasting of climate phenomena

- focus on extreme events e.g. covid
- "The dependence on precise initial and boundary conditions and the necessity to simplify, inherent to any modeling approach, as well as the chaotic nature of the system under study will hit hard limits to further improvement [of numeric models]"
- not all processes resolved in numerical models, no teleconnections etc.
- finding couplings might not lead to better single-point prediction, but information is valuable
- similarity quantified with different approaches (Pearson, CondMutInf, TrasnferEntropy, ParCorr, Granger)
- surrogate data: shuffled versions of the original time series or synthetic time series that match the relevant statistical properties
- measurement errors play a much smaller role than for numerical models
- complex networks provide ideal tools for data exploration
- examples: el-nino early warning sign is heigh connectivity of network
- complex correlation networks provide a more explorative approach, helping to detect patterns in large high-dimensional data, which can give rise to new hypotheses, which could, in turn, be tested with the PCMCI approach.

## Reconstructing regime-dependent causal relationships from observational time series

- Regime-PCMCI: PCMCI followed by regime learning linear optimization approach
- applied to El-Nino and Indian rainfall
- regime-dependent autoregressive models (RAMs) -> need seasonal index or similar (regime is known)
- explore "Markov-switching ansatz" with PCMCI
- iteratively refine regime variable Gamma and causal graph
- Fixing one variable to estimate the other allows in both cases to solve the individual optimization step via linear programming
- doesn't work well if only slight change in causal effect or in time lag (e.g. Y-1 => X to Y-2 => X)

## Introduction to Focus Issue: Causation inference and information flow in dynamical systems: Theory and applications

- summary of multiple articles!
- information theoretic: Shannon entropy, mutual information, transfer entropy
- in linear gaussian case: Transfer Entropy == Granger causality
- causation -> dynamical nonlinear attractors via time-delay embedding (takens theorem?) "closeness principle"
- optimal causation entropy (oCSE) as alternative to PCMCI
  **PAPERS:**
- "Detecting directional couplings from multivariate flows by the joint distance distribution"
- information flow from macro- to micro-scale is adequately captured by transfer entropy, and no synergistic effects are present
- simulate social media posts and quoting --
  J. Bagrow and L. Mitchell “The quoter model: A paradigmatic model of the social flow of written information”

## Detecting Causality in Complex Ecosystems

- invents "CONVERGENT CROSS MAPPING"
- in granger causality for deterministic systems if X is cause for Y the residual will still be correlating -> nonseparability
- ecosystems: weak/moderate coupling is norm, external forcings (e.g. temperature) make for spurious correlations, system is "non-separable"
- bivariate?
- the idea is to see whether the time indices of nearby points on the Y manifold can be used to identify nearby points on MX
- only estimates states (eliminates information loss from chatic dynamics)
- convergence: cross-mapped estimates improve with time series length -> for real data: predictability increases

## DAGs with NO TEARS: continuous optimization for structure learning

- learning DAGs from data is NP-hard
- score-based, turn combinatorial into continuous program
- adjacency matrix
- use h: smooth function with computable derivatives over adjacency matrix for acyclicity with h(W) = 0 if W is acyclic
- for undirected graphs: "log-det" program
- program is "nonconvex"
- focus on linear structural equations and use "least-squares" LS loss
- faithfullness assumption not required
- main problem: #acyclic graphs superexponential
- h(W) = tr(e^{W°W}-d) = 0 -> matrix W is DAG
  "°" is Hadamard product, e^A is matrix exponentail of A
  gradient of h: Delta h(W) = e^{W°W}^T ° 2W
- faster and better than greedy (score-based) search

## Review of Causal Discovery Methods Based on Graphical Models

- PC and FCI (Fast Causal Inference) constraint-based: no confounder for PC, FCI with confounders - output MEC
- GES fGES score-based: no confounders - add edge that most improves fit
- Functional Causal Model (FCM) based: Lingam - Central limit Theorem: sum of ind var gets close to gaussian but never exactly
- statistical estimation: "consistency"- convergence to true value, "uniform convergence" probabilistic bounds of errors

## Causal Discovery using Marginal Likelihood

- like LiNGAM - using independent mechanisms assumption to infer direction

## Causal Adversarial Network for Learning Conditional and Interventional Distributions

- Label Generation Network LGN + Conditional Image Generation Network CIGN
- genrate samples from interventional distributions
- can generate interventional+conditional samples without access to causal graph
- task: class conditional image generation
- WGAN-GP: uses Wasserstein distance (earth mover), GP: enforces Lipschitz constraint by adding gradient penalty term
- use adjacency matrix
- uses acyclicity constraint for optimization (trace of matrix squared thing)
- mask vector to decide whether var is conditioned on or uses SCM
- conditional relations between labels and pixels and between pixels
- since assumption: label -> image : conditioning same as intervening (fixing)

## CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models

- VAE: learns a probabilistic model of the data
- The framework of variational autoencoder (VAE) is commonly used to disentangle independent factors from observations
- CausalVAE model is able to generate counterfactual data through “do-operation" to the causal factors
- add Structural Causal Model layer (DAG) to NN -> after encoder, decoder reconstructs input through causal model
- Disentangled Representation Learning: learn mutually independent latent factors
- example using shadow of pendulum with causes: pos of pendulum and light
- intervening on faces (e.g. mouth open, gender, smile)

## Conditional Adversarial Debiasing: Towards Learning Unbiased Classifiers from Biased Data

- replacing normal independence test debiasing with conditional independence on label
- example: aquaplaning conditions have different background for safety reasons
- counterexample: missing from work, pregnant, female
- separate nn between feature extractor and predictor
- other example: green/red crosses or rectangles
- better than other methods of debiasing

## BIC - Bayesian Information Criterion

- can be used to find the best model
- log likelihood L
- number of parameters k
- number of samples used for fitting n
- AIC = 2k - 2L
- BIC = k\*log(n) - 2L
- train on as few numbers as possible, with few parameters and largest log likelihood

## Learning with Kernels - A Tutorial Introduction

- kernel ~ similarity measure
- first embed data into feature space H via map phi
- simple learning algorithm: compute mean of 2 classes, find middle, find vector connecting sample and middle, check if angle smaller pi/2 -> süecial case: Bayesian classifier
- support vectors: subset of training patterns (remove patterns far from decision boundaries)
- we minimize _training error_ = _empirical risk_ does not necessarily imply small _test error_ = _risk_
- VC bound (Vapnik-Chervonenkis)
- use Lagrangian for optimization
- support vectors have non zero lagrangian weights a_i
- SVM: express dot product <x,x'> in terms of kernel k(x,x') = _kernel trick_ maximize margin to datapoints while minimizing classification error
- if data not linearly separable: map into higher dimensional feature space
- Kernel Principal COmponent Analysis can be done
- popular kernel: Gaussian radial basis function (RBF) - computes similarity based on euclidean distance

## VQ-VAEs: Neural Discrete Representation Learning

- VAEs but instead of continous sampling (latent) space use discrete space - code book
- code book: set of vectors - find closest code book vector using L2 norm
- VQ = Vector Quantizer

## Bernhard Schölkopf: Learning Causal Mechanisms (youtube)

- independence of mechanisms
- causal and anticausal machine learning models
- adversarial noise -> might lead to completely different results - interventions

## Entropy-Based Discovery of Summary Causal Graphs in Time Series

- method similar to PCMCI
- new measure "causal temporal mutual information measure" for pcmci/fci like methods

## Causality and Graph Neural Networks (YouTube)

- knowledge graphs: e.g. recommender systems: why did person buy thing

## Conditional independence testing based on a nearest-neighbor estimator of conditional mutual information

- CMIknn: computing conditional mutual information using k nearest neighbours

## Finding the right XAI method — A Guide for the Evaluation and Ranking of Explainable AI Methods in Climate Science

- Problem: no ground-truth explanation
- properties: robustness, faithfulness, randomization, complexity, and localization
- Integrated Gradients, Layer-wise relevance propagation, and InputGradients: robustness, faithfulness, complexity but no randomization
- Gradient, SmoothGrad, NoiseGrad, and Fusion-Grad: opposite
- 3 categories: local and global decision-making as well as self-explaining models
- Local explanations: e.g. pixel importance
- global explanations: showing a map of important features for whole class
- Self-explaining models: additional "transparent" prototype layer after the convolutional layers
- model-aware and model-agnostic methods
- climate research mostly local explanation methods
- XAI evaluation: metrics for measuring reliability
- compare:
  - "Gradient/Saliency", change in prediction given change in features
  - "InputGradient", product of gradient and input -> higher score if in data
  - "Integrated Gradients", introduces "mean" image-> diff to that
  - "Layerwise Relevance Propagation (LRP)", backprop prediction with specific rules
  - "Smooth Grad", add rand noise to input, average explanation
  - "NoiseGrad", perturb model weights
  - "FusionGrad", perturb input and weights
- metrics:
  - Robustness: Local Lipschitz estimate, Avg-sensitivity -> use Monte Carlo
  - Faithfulness: changing features with high relevance greater effect than low -> apply Remove and Debias (ROAD) -> mask important features
  - Randomisation: model parameter test (MPT) corr between expl and model or rand model, random logit (RL): similarity between heatmap explaining wrong label and correct
  - Localisation: high relevance of ROIs (user defined regions of interest) -> "top-k-pixel"
  - Complexity: concise, few strong features -> low entropy, high Gini index
-

Mamalakis 2021, 2022
Labe and Barnes, 2021
Montavon et al., 2019
Alvarez-Melis and Jaakkola, 2018
Yeh et al., 2019
Rong et al., 2022a
Hurley and Rickard, 2009 -> gini index

## Using Causal Inference to Globally Understand Black Box Predictors beyond Saliency Maps

- previous methods:

1. only local explanation (true for one input or inputs close to it)
2. only assigns saliency to part of input, not aggregate functions of input
3. don't handle confounding

- no information on model -> black box estimator
- for feature visualization maximizes response of neuron, but function may not be linear (-> multiple maxima)
- saliency map 3 methods: gradient of loss function depending on input, taylor-approximate learned function (evaluate as polynomial on point), substitue part of input with neutral alternative and record change (problem: what is "neutral")
- construct causal graph with GT (ground truth), X (feature), -X(other f), P (prediction), TS (training set), W (weights): Is X independent of P given GT ?
- toy dataset with 8x8 pixels tested mean/variance estimator against NN
- very limited especially because of strong assumptions, cond ind tests, only tested on simple toy model

## Causes and Explanations: A Structural-Model Approach. Part I: Causes - Pearl

- Actual Causes?
- Hume: "We may define a cause to be an object followed by another, ..., where, if the first object had not been, the second never had existed"
- if fire A had not started, fire B would be cause and burned down house should not be "actual cause"
- necessary causality is related to but different from actual causality
- related to specific events: was X = x the actual cause of Y = y in that scenario?
- lots of interesting examples showing how the choice of the SCM is very important

## Causes and Explanations: A Structural-Model Approach. Part II: Explanations - Pearl

- The basic idea is that an explanation is a fact that is not known for certain but, if found to be true, would constitute an actual cause of the explanandum (the fact to be explained), regardless of the agent’s initial uncertainty
- the definition of an explanation should be relative to the agent’s epistemic state
- if A is explanation of B happening, after "discovering" A it is not an explanation anymore but part of epistemic state
- we have disallowed disjunctive explanations: there could be many reasons at once
- better explanation: has higher probability of occuring
- partial explanations: victoria went to the canary islands and it was sunny there
- dominant in AI: maximum a posteriori (MAP) approach: best explanation for an observation is the state of the world that is most probable given the evidence

## Determining the Relevance of Features for Deep Neural Networks

- often used correlation between feature and output -> problems: nn non-linear, confounding
- to mitigate problem of confounding: gradient between input and output instead of correlation -> only single examples
- we cannot distinguish wether function is f(x) or f(x³), but it doesn't matter
- trained on real life image classification cnn -> testing "global" property _area of object_ (as fraction of image pixels based on ground truth)
- Note that a saliency map only highlights the region in the image that contains, for example, the yellow lores, while our method can distinguish between the area and the color of this region as different features
- feature _symmetry_ of a skin lesion
- "feature of interest" is chosen and described manually
- treat model as black box -> only look at dependence between feature and model output

## Persuasive Contrastive Explanations for Bayesian Networks

- different categories of explanation (of bayesian networks): explanation of ...
- evidence
- reasoning: justification for the obtained outcomes and the underlying inference process, could also address the explanation of outcomes _not_ obtained (_contrastive_)
- the model itself
- decisions
- XAI also important for bayesian networks
- _explanation context_: tuple <**e**,t,t'> where t = T(T|**e**) and t != t' € O(T)
- _contrastive explanation_: combines a sufficient explanation for t with a counterfactual explanation for t'
- _persuasive contrastive explanation_: pair of evidence subsets [s,c]: s = minimal subset that outputs t, c = min subset for which T(T|e'c) = t'
- probabilities Pr(s~e') and Pr(e'c) > 0 (impossible outcomes are senseless)
- sufficient explanation: subconfiguration that gives same outcome regardless of rest of evidence
- proof: multiple counterfactual explanations if not all var binary-valued, otherwise unique
- counterfactual set must includes at least one variable from sufficient set
- find all sufficient + counterfactual explanations: organize search space using "annotated lattice"

## Counterfactuals uncover the modular structure of deep generative models

- previous work: use independence to "disentangle" latent factors
- too restrictive: counterfactural manipulations -> can be used to design targeted interventions (i.e. style transfer of images)
- state-of-the-art: GAN and VAE
- explore modularity using concept of Independent Mechanisms
- classical notion of disentangled representation: individual latent variables “sparsely encode real-world transformations”
- unsupervised: statistical notion: conditional independence of latent factors
- problems:
  - statistical independence of true data generating mechanisms unlikely (e.g. hair-skincolor)
  - independence constraints not enough
  - on real world data sets disentangled gan much worse than non-disentangled
- **Intrinsic disentanglement**
- make heatmap with something like "average treatment effect" (expectation of difference in output )
- fine to coarse: individual channels: _elementary influence maps (EIM)_
- cluster EIMs: smoothing, thresholding, non-negative matrix factorization -> S= WH, cluster based on maximum weight

## Towards robust explanations for deep neural networks

- derive bounds for maximal manipulability
- other methods are unreliable because of adversarial attacks [31]
- input manipulation: slightly changing picture, model manipulation: same output but different explanation, but here input
- 3 techniques: weight decay, smooth activation function, minimize Hessian of network
- construct "curve" between unperturbed x and adversarially perturbed x_adv
- intermediate points in practice steps of optimization
- Frobenius norm of change of Hessian of j-th component of explanation is bounded -> maximal change bounded
- frobenius norm ~ l2 norm of matrix (distance)
- Hessian (Hesse-Matrix): ~ analog zur 2. ableitung, Jacobi-Matrix ~ 1. Ableitung
- method: add frobenius norm of hessian to loss function
- actual computation too expensive: estimate stochastically with mini batches
- minimizing frobenius norm -> use weight decay (well established as generalization method)
- rather use softmax+ than ReLU activation function as ReLU'' ill-defined (but can be shown that method also works for ReLU)
- experimental analysis: use different similarity scores to compare visual output of explanations
- restrict to small noise: better accuracy, worse explanations
- taylor series: f and infinite sum of derivatives are ~equal at x

## Unmasking Clever Hans predictors and assessing what machines really learn

- standard performance evaluation metrics bad at distinguishing short-sighted and strategic problem solving behaviors
- use SPectral Relevance Analysis
- is some sort of semi-automation using LRP
- SpRAy: compute relevance maps (LRP), eigenvalue based spectral clustering -> DR using t-SNE

## Deep Taylor Decomposition - Gregoire Montavon - Video

- simple taylor decomposition not good enough
- do as many taylor expansions as there are neurons -> n neurons -> n ableitungen
- search root point along specific direction (root point for taylor decomposition)
- direction: a_i\*1_w'\_ij>0: stays within input domain,
- different rulse for different kinds of layers
- deep taylor decomposition good for finding appropriate rule for input domain

## Attention is all you need - Seminal Paper introducing Transformers (gpt)

- multihead attention
- defines what attention is (complicated matrix stuff)
- is used in modern LLM like chatgpt

## Information transfer in social media

- exploring transfer entropy to measure influence in social media
- if # url first tweeted by A and subsequently by B then transfer entropy is high
- problem: how to discretize events: human reaction times have strong tail
- identified spam accounts very well
- politician has stronger influence on followers behavior as Souljaboy

## A measure of individual role in collective dynamics

- complex network representations: good for heterogeneousdynamics, few nodes important
- show that _dynamical influence_ is good centrality measure quantifying how a nodes dynamical state affects collective behavior
- applied to "Kuramoto model" and "Roessler Chaotic Dynamics"
- classical measures: centrality, betweenness, eigenvector...
- recent:laplacian-based related to PageRank
- "leading left eigenvector of a characteristic matrix that encodes the interplay between topology and dynamics"
- N time-dependent real variables (x1...xn) coupled in NxN matrix
- consider largest eigenvalue of M (my_max), if my_max < 0 x(t) converges to null vector with stable fixed point solution
- if my_max > 0, x(t) grows indefinitely
- if my_max = 0 exists, then scalar product between left eigenvector of M for my and x is conserved quantity -> can compute x(inf) as a limit
- example: _susceptible, infected, removed_ panemic spreading process
- _Ising model_ binary state model, zero temperature version -> majority rule for state updating, needs noise
- there: spreading efficiency is correlation between state of i at time t and "magnetization" of whole system at later stage
- for _voter model_ my_max = 0 - diffusion process

## Social tipping processes towards climate action: A conceptual framework

- in comparison to more "natural" tippin processes: human agency, social-institutional network, different spatial/temporal scales, increased complexity
- example: european political system and fridaysForFuture
- examples for social tipping processes: divestment, political mobilization, social norm change, socio-technical innovation
- if no action _undesirable_ social tipping points, e.g. mass migration, food system collapse, revolutions
- related fields: (1) ecology and social-ecological systems research, (2) climate change science, (3) theories of social change involving threshold phenomena, and (4) sustainability science with a focus on transitions and transformations
- interaction of ecological and social causes "co-determinants"
- social tipping processes resaerch is much older (e.g. racial segregation of neighbourhoods)
- associated with "critical mass phenomenon"
- this and current research mostly about how to create desirable tipping points
- DEF: “the point or threshold at which small quantitative changes in the system trigger a non-linear change process that is driven by system-internal feedback mechanisms and inevitably leads to a qualitatively different state of the system, which is often irreversible.”
- 25 experts made definitions in workshop
- LOL: "While humans have a generally poor track record of utilizing their agentic capacities especially with regard to shaping the future (e.g. Bandura, 2006; European Environmental Agency, 2001; 2013), they appear unique in their capacity to transcend current realities with their decisions
- "ability to anticipate and imagine futures enables humans to transcend the present shape of the future accodring to values and goals"
- agency creates structures/networks, the in turn enable and/or constrain agency
- social networks: links are more diverse than in other networks
- through new technology overall interaction and spreading rates increase dramatically - affecting stability
- tipping: more like infection, or changing structure of network (e.g. polarization)
- social tipping typically faster than climate tipping
- macro-, meso-, micro-levels or nested
- _spatial-temporal ephemerality_: can appear/disappear _out of nowwhere_
- "This kind of causality – multiple interacting, distributed causes across varying scales – are a key characteristic of complex systems"
- hypothesis: societies are somewhat irreversible (e.g. no return to monarchy)
- _Intervention Time Horizon_ decision+actions between now and horizion influence whether system tips -> not more than decades, not equal to "planning time horizons", which can be longer
- _Ethical Time Horizon_ consequence too far in future not relevant
- important to identify _critical states_

## Towards representing human behavior and decision making in Earth system models – an overview of techniques and approaches

- summarizing different ways to model human behavior + decision making and interaction in a dynamic system
- _decision making_ cognitive process of delibreately choosing between alternative actions, my involve analytic and intuitive modes of thinking
- _actions_ intentional and subjectively meaningful activities of an agent
- _Behavior_ broader concept that also includes unconscious, automatic activities, habits+reflexes
- (1) model categories, (2) modeling approaches and techniques, (3) important considerations for model choice and assumptions
- (1) (A) individual decision making and behavior, (B) interaction between individuals, (C) Aggregation and System-Level Description
- (2) (A) rational choice, decision trees, learning theory, (B) game theory, social influence models, networks, (C) voting, general equilibrium, agent-based, statistical, system-level
- (3) (A) motives, constraints, knowledge, strategy, (B) strategy, imitation, influence on beliefs, opinions, (C) homo-/heterogeneity, feedbacks, transient/stable states, centralization
- backward, forward and sideward looking behavior
- (computational) learning theory as a social modeling technique?

## Anticipation-induced social tipping: can the environment be stabilised by social dynamics?

- "model socio-ecological coevolution with specific construct of _anticipation_ acting as a mediator between social and natural system"
- agent-based, but appeoximating microscopic behavior
- 3 contributions to probability of stopping pollution: direct environmental impacts, social contagion, anticipated impacts

## The challenges of integrating explainable artificial intelligence into GeoAI

- geospatial dimensions -> spatial scaling, handle topology (not just like heatmaps on images)
- ethical implications, e.g. classification of neighbourhoods and racism
- limitations because of cultural differences

## Towards Neuro-Causality - Relating Graph Neural Networks to Structural Causal Models

- special interventional graph neural network, not counterfactual
  Derrow-Pinion et al., 2021 - google maps
  Mitrovic et al., 2020 - represenation learning causal

## Knowledge graph-based rich and confidentiality preserving Explainable Artificial Intelligence (XAI)

- semantic technology
- about demand forecasting in industry 4.0
- modular architecture: semantic explanations, influencable factors, context, opportunities for data enrichment
- ontology to capture demand forecasting domain knowledge
- use data like unemployment rate, inflation, income gpd...

## Information fusion as an integrative cross-cutting enabler to achieve robust, explainable, and trustworthy medical artificial intelligence

- Robust AI solutions must be able to cope with imprecision, missing and incorrect information, and explain both the result and the process of how it was obtained to a medical expert
  **1. Complex Networks and their Inference**
- complex diseases -> graph vis, DR, effective models to provide _relevant systems view_
- knowledge repositories
- quantivative graph theory -> Comparative Network Analysis, Network Characterization and networks explainable by design
  **2. Graph causal models and counterfactuals**
- occams razor: (if it can be explained by one cause same as 2, its the one)
- questions: what is p that x causes y, what is p that relationship exists between x and y
- (nearest) counterfactual explanations
- communicate uncertainty -> when is data not in same iid as training data
- contrastive explanations
- _causal AI in medicine is how to disentangle correlated factors of influence in high-dimensional settings_
- apply learning approaches to the discovery of counterfactuals
- main concern for e.g. melanom detection: low false negative rate
- interventional clinical studies can be driven by the results of causal analysis
- if counterfactual is close to patients case, AI not to be trusted
  **3. Verification and Explainability methods**
- needs to be verifiable and explainable for legal reasons
- human-in-the-loop approach
- explainability extended to _causability_: " causability measures whether an explanation achieves a given level of causal understanding for a human"
- current systems lack especially **robustness**
- especially problematic as we gain more insights into diseases, change treatment etc
- concepts: _transfer learning_, _domain adaptation_, _adversarial training_, _lifelong/continual learning_
- grand issue: estimate generalization error -> e.g. with complexity measures (Vapnik-Chervonenkis dimension, stability)
  In general:
- Problems: no generalization, need huge data, sensitive to perturbation, especially: no causal relationships
- AI + experts + data fuse -> complex networks and their inference -> craph causal models -> verification and explainability
- "omics" = study of biological process on the molecular level (e.g. genomics, proteomics...)
- Big problem: confounding and selection biases -> assumptions with a priori domain knowledge
- therefore: **linchpins** -> (1) target trials, (2) transportability, and (3) prediction invariance
- (1) target trials: algorithmic emulation of randomized studies
- (2) transportability: _license_ to transfer causal effects to different iid
- (3) prediction invariance: if accuracy does not vary across settings -> true causal model underlying
- robustness must be verified especially for: _adversarial attakcs, counterfactual explanations, inherently biased input_
- adversarial training improves interpretability of saliency maps
- technical advantages: bias identification, adversarial attack detection, reducing model complexity/size
- medical expert advantages: increased trust, remaining responsibility, avoidance of discrimination
- _Mutual explanation_ between medical expert and her AI
- language is very important: knowledge + explanations channel
- need cross-modal (e.g. image-text-omics) representations
- also mention attention scores: only loosely correspon to human-acceptable explanations
- using AI to fuse data sources in the internet reliably

## A Unified Approach to Interpreting Model Predictions (SHAP values)

1. view explanation as a model itself **explanation model** -> class of _additive feature attribution methods_
2. game theory results guaranteeing a unique solution apply to entire class -> SHAP
3. SHAP value estimation methods, better aligned with human intuition

- local methods explaining f(x) (x is single input) with explanation model g(x)
- try to ensure that _local_ model also approximates other inputs
- LIME: local linear explanation -> compute using penalized linear regression
- DeepLIFT: effect of input feature being set to reference value
- Layer-Wise Relevance Propagation = DeepLIFT with reference activation 0
- Classic Shapley Value Estimation: 3 prev methods:
  -- shapley regression values: linear models for all feature subsets -> values are weighted average of all possible differences of subsets with/without curr feature
  -- shapley sampling values: just sample from previous
  -- quantitative input influence: also similar, but broader framework
- desirable properties:

1. Local accuracy -> approximation matches output at least for specific input
2. missingness -> if simplified model misses feature, it has no impact
3. consistency -> if inputs contribution >= regardless of other features, attribution should not decrease

- SHAP: change in expected model prediction when conditioning on feature
- model agnostic approximation: shapley sampling values, Kernel SHAP
- model-specific approximation: for linear model just from weights,

## Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models

- extend shapley values with causal graph somehow
- even counterfactual explanations make simplifying assumption that features are independent
- -> conditional shapley values: conditioning by intervention (do-calculus)
- others argue: interventional shapley values simplify to marginal ones in ML algorithms
- separate total causal effect into direct and indirect
- SHAP=attribution method: attribute difference between f(x) and baseline f0 to features:
- f(x) = f0 + sum psi_i, where psi_i is contribution of feature i, f0 is expeceted average
- assign feature values in order pi, contribution of i given pi = difference in value function _v_
- _v_(S) = E[f(X) | do(X_S = x_S) (S is subset of "in coalition" indices with known feature values x_S)]
- make no distinction between real world and inputs of prediction model
- **classical norm theory** It states that humans, when asked for an explanation of an effect, contrast the actual observation with a counterfactual, more normal alternative
- "humans sample over different possible scenarios here: different orderings of the features"
- asymmetric (like temporal chain) -> blame foremost the root cause
- asymmetric shapley values don't work for XOR function (faithfulness?)
- asymmetric better for temporal order, symmetric less sensitive to model misspecifications
- if only partial causal graph -> "causal chain graph" with directed and undirected edges -> MAG
- causal or conditional shapley values way more computationally expensive

## The Causal-Neural Connection: Expressiveness, Learnability, and Inference

- neural nets not in theoretical limit able to determine SCMs
- an arbitrarily complex and expressive neural net is unable to predict the effects of interventions given observational data alone
- it is still largely unknown how to perform the tasks of causal identification and estimation in arbitrary settings using neural networks as a generative model, acting as a proxy for the true SCM M∗
  Contributions:

1. special SCM which can use gradient descent -> NCM neural causal model, prove expressiveness, ability to encode inductive bias
2. formalize "neural identification" prove duality of identification in SCM vs NCM -> sound and complete algorithm to jointly train NCM
3. gradient descent algorithm to estimate causal effects

- aim to resolve tension between _expressiveness_ and _learnability_
  **NCM**: parameters \theta = {\theta_v_i: V_i \in V} is an SCM <U,V,F,P(U)> where
- U -> unobserved variables/confounding
- F -> feedfoward NN for each variable parameterized by \theta_v_i mapping U_V_i and Pa_V_i to V_i
- P(U) ~ uniform distribution (0,1)
  Remarks:
- each NCM is SCM, but not each SCM is NCM (U not always uniform dist, F not always feed-forward NN)
- non-markovian -> variables might share unobserved confounder
- feedforward nets are universal approximators, any probdist can be generated by uniform one -> may be expressive enough
  THEOREM 1: for every SCM M* \exists M so that it is L3 consistent to M* -> see appendix C.1
  COROLLARY: but expressiveness does not mean that the learned object has the same empirical content as the generating model
- _idea_ multiple feedforward NNs might produce same observational distribution, but not same interventional distributions
- "solution": narrow down class of M\* with constraints:
- causal bayesian network (CBN), G-consistent SCM M, if causal bayesian network G is a CBN for L_2(M)
- if G known, only consider NCMs that are G-consistent -> algorithm for constructing such NCM
- C^2 component: fully (bidirectionally) connected subgraphs of SCM
- G-constrained NCM: unobserved confounder of variables only if those var form C^2-component, parent only if directed edge
- algo1: all and only identifiable effects are found -> theoretically as powerful as do-calculus: if min/max of parameter match=identifiable
- algo2: minimize and maximize interventional distribution, if max-min < threshold -> is identifiable, is effect

## CITRIS: Causal Identifiability from Temporal Intervened Sequences

- a variational autoencoder framework that learns causal representations from temporal sequences of images in which underlying causal factors have possibly been intervened upon
- learns interventions such as 3D rotation angles
- instead of position as x,y,z and rotation too, create multidimensional macrovariables
- causal factor potentially multidimensional vector
- assume that intervention targets observable: occurs for instance when we have access to a set of observation and action trajectories. We refer to this setup as TempoRal Intervened Sequences (TRIS)
- availability of intervention targets: binary vector I^t where I^t_i = 1 -> causal variable C^t_i was intervened upon
- cannot disentangle 2 causal factors if they are never intervened, or always intervened jointly
- for multidimensional causes even worse, because some dimensions may always be unaffected by given interventions
- therefore _minimal causal variables_: find split that splits domain into variable and invariant part
- split must be independent of intervention variable, given parents of its factors
- _minimal causal split_: maximizes information content (entropy H) of invariant set given parents of causal factor C_i
- H = entropy in discrete case = limiting density of discrete points (LDDP)
- causal graph may differ from true SCM: may only have parents that are influenced by intervention

## Towards Causal Representation Learning

paper we read in causal inference class
all current research fields:

- prediction in i.i.d setting: prediction from anti-causal/non-causal features not desirable, not robust to interventions
- under distribution shifts: generalizability, robustness of ML approaches
- answering counterfactual questions, critical in reinforcement learning
- independent causal mechanisms: functions do not inform each other
- sparse mechanism shift: small distribution changes manifest in sparse/local way in causal factorization
- causal discovery: cond ind testing difficult, needs lots of data, 2 variables -> not identifiable -> restrict function class
- learning causal variables = causal representation learning
- preventing adversarial attacks
- pretrain on huge dataset, improve e.g. with selfsupervised learning
- use data augmentation (e.g. scaling, rotation) to enrich dataset and make robust to interventions
- reinforcement learning is close to causality: sometimes estimates do-probabilites

## Weakly supervised causal representation learning

- very similar to CITRIS, same authors -> have some pairs of samples before/after intervention
- really cool example application: robotics: image interventions (e.g. light source position, color, scaling, rotation...)

## A Symbolic Approach to Explaining Bayesian Network Classifiers

- explainability methods for bayesian classifiers
- two types of explanations:

1. minimal set of currently active features
2. minimal set of features (active or not), whose state is sufficient for classification

## Selecting Robust Features for Machine Learning Applications Using Multidata Causal Discovery

- use PCMCI and PC1 to identify important features for prediction (PC1 = only 1st step of PCMCI)
- application case: predicting intensity of Western Pacific Tropical Cyclones
- multidata -> more than one dataset, ensemble of time series, but bompute PC1/PCMCI for all at once
- important tasks for environmental applications: DR, good feature selection for cheaper computation, generalization, explanation

## Backtracking Counterfactuals

- based on human intuition: first cause is true cause -> backtrack to altered initial conditions instead of slightly changing SCM (pearl: intervening)
- general formal semantics for backtracking counterfactuals within SCMs
- interventional approach is purely forward-tracking: setting e.g. Y=3 only has effect on descendants of Y
- new approach: backtracking all changes to changes in values of exogenous variables (U)
- e.g. not setting Y* = 3 but Y* = U_y* + X* = 3 -> infinite possibilities to split
- explain entirely by change in U_y*, or X*, or half-half
- to pick: notion of preference -> closeness/similarity measure
- _backtracking conditional_ P_B(U\*|U) - likelihood of counterfactual world
- which variables / U*i are more likely to change (how do standard deviations compare) - given the \_backtracking conditional* distribution
- properties of this distribution:

1. preference for closeness: U\* should be close to given U, distribution assigns higher probability to values close to given values
2. symmetry: P_B(u*|u) = P_B(u|u*)
3. decomposability: since exogenous variables in U independent P_B(u\*|u) should factorize completely

- use distance function to construct this backtracking conditional
- for U \in \R -> (squared) Mahalanobis distance: d(u*,u) = 1/2(u-u*)^T sum ^-1 (u-u\*)
- other possibility: completely dismiss what happened and just take prior probability distribution of U
- for _observational_ (backtracking) counterfactuals **F** does not need to be completely known

## Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications (2020)

- not just accuracy but also: robustness, resilience to drifting data distributions, assess confidence, safety, legal requirements, complement humans, reveal interesting correlations, protection against spurious correlations,
- focused on _post-hoc_ explanation methods: take any model and analyze after training
- global: "prototypes" of a class
- local: decision function can be linearly approximated locally
  Four families of techniques:

1. **interpretable local surrogates**

- LIME (linear local approximation)/ decision trees
- LORE
- Anchors

2. **occlusion analysis**

- leaving out patches of input -> build heatmap of strongest decrease
- shapley values: not just one feature -> with conditionality e.g.
- meaningful perturbation: synthesize occluding pattern with maximum drop -> also: rate distortion theoretical framework

3. **gradient-based techniques**

- integrated gradients
- SmoothGrad

4. **layerwise relevance propagation**

- output score is progressively redistributed backwards till input

5. **others**

- graph neural networks: show graph at every layer
- generative approaches, e.g. structured textual explanation
- BiLRP: tracking parts that are similar in 2 images

6. **global explanation methods**

- self-explainable models: linear, nn with explicit top-level sumpooling structure, attention mechanisms
- TCAV -> explain through latent space
- relevance pooling: which features are on average important
- spectral relevance analysis (SpRAy): cluster a few prototypical explanations
  **Desiderata for evaluation of explanations:**

1. faithfulness/sufficiency

- pixel flipping: test if removing features marked as important leads to strong decay of prediction abilities

2. human-interpretability

- associated file size: - compress heatmap image, if filesize low, interpretable

3. practical applicability (e.b. algorithmic efficiency)
   **Theoretical Foundations**
1. Shapley Values: consider every subset of features, what is payoff of adding feature i in expectation over all subsets
1. Taylor decomposition: approximating function through first few terms of taylor series (derivatives)
1. Deep Taylor decomposition: perform taylor decomposition at each layer of dnn
   **Analysis of large number of heatmaps:**

- first goal: find data artifacts - clever hans effects
- second goal: investigate learning model to find novel prediction strategies

## Bayesian learning of Causal Structure and Mechanisms with GFlowNets and Variational Bayes

- _Bayesian Causal Structure Learning_: learn posterior distribution over DAG
- bayesian -> can reason about uncertainty of causal model, good because could be unidentifiable from observational data
- method: **"Variational Bayes-DAG-GFlowNet"**
- def. of causal bayesian network (causal model): "a family of distributions with shared parameters, corresponding to all possible interventions on the variables"
- recent bayesian causal structure learning uses gradient descent methods
- new: not only infers graph structure but also parameters of linear gaussian model, guarantees acyclicity
- is more of a _score-based_ method (i.e. defining score how good DAG is, search over possible DAGs, e.g. Bayesian Gaussian equivalent score)
  **Related Work:**
- compare to NO TEARS framework: soft constraint h(W) (differentiable) function expresses DAG-ness - search over continous space
- MCMC: monte carlo markov chain/ sampling approach (markov property: only need to know direct parent)
- variational inference/parameterized distribution: find distribution by fitting function to posterior by minimzing KL divergence
  = q\*(Z) = argmin KL (q(Z) || p(Z|X=D)) (over family of distributions, e.g. gaussians) -> rearange, because we only have joint distribution p(Z,D)
  -> ELBO (evidence lower bound) = L(q) = E\_{T~q(Z)}[p(Z,D)/q(Z)] is negative -> maximize ELBO
- GFlowNet: generative, produces policy, with probability proportional to reward function R(x)
- policy: transitions with probability between all possibilte states, states form a DAG through which "samples" flow
- transitions parametrized by flow functions
- actions: sequentially either add edge or stop
- did not perform better than other tools (e.g. GES-greedy equivalence search)

## Attention is not Explanation

- claim: attention weights do not provide meaningful explanations
- adversarially look at attention weights
- properties shoud be:
  (i) attention weights correlate with feature importance measures
  (ii) counterfactual attention weights should yield different prediction or be equally plausible
  Questions:

1. specifically: "do attention weights indicate _why_ a model made a prediction -> _faithful_ explanations"
2. correlation with gradients and "leave-one-out" feature importance methods
3. alternative attention weights -> different predictions?

- works better in simple feed-forward encoders
- random attention weights fail harder if few tokens responsible for positive class (e.g. "diabetes")

## Attention is not not Explanation

- answer to previous paper
- test contribution of attention weights by setting to uniform distribution
- expected variance when initializing randomly -> better interpret adversarial results
- showing that using trained attention weights on simple multi-layered perceptron (MLP) has skill
- model-consistent adversarial training
- adversarially trained models don't perform well as weights to MLP
- claim "Attention Distribtuion is not a Primitive" -> base attention weights are not assigned arbitrarily and fit with rest of model
- _an_ explanation and not _the_ explanation
- by proving that random attention works, have not proven that such a model could be trained if linked with other parameters

## A causal framework for explaining the predictions of black-box sequence-to-sequence models

- perturbing input to get input-output pairs, generate graph over tokens
- basically like attention too

## From “Where” to “What”: Towards Human-Understandable Explanations through Concept Relevance Propagation

- local vs global explanations
- global: _what concepts a model has generally learned to encode_
- **concept relevance propagation**: Glocal XAI can identify the relevant neurons for a
  particular prediction (property of local XAI) and then visualize the concepts these neurons encode
  (property of global XAI)
- _relevance maximization_ find representative samples of concept, based on _usefullness_
- local: superposition of many different model-internal decision sub-processes
- local attribution map, loses information: e.g. where in image but not what was important
- global: synthesize example data, which neuron activates for, but not how features interact
- with CRP can:
  (1) gain insights into the representation and composition of concepts in the model as well as quantitatively investigate their role in prediction
  (2) identify and counteract Clever Hans filters focusing on spurious correlations in the data
  (3) analyze whole concept subspaces and their contributions to fine-grained decision making
- problem: attribution scores for hidden neurons are not interpretable
- normal LRP: can do class- or output-conditional relevance map given sample x, to have heatmap for each class
- sometimes class-specific heatmaps can help, but other time not (e.g. bird species)
- class- or output-conditional relevance quantity R(x|y) by masking unwanted network outputs !y
- conditioning set theta, might apply to multiple hidden layers
- for convolution: each kernel channel encodes one concept,
- for fully connected: each neuron one concept
- **Hierarchical Composition of Relevant Concepts**: when interesting concept c found, backpropagate further to lower layers to find what sub-concepts it is composed of
- **activation maximization** find "image" that maximizes activation of a neuron for sampling (e.g. using Gradient Ascent)
- generated sample often not interpretable, therefore data-based approach
- relevance-based reference sample selection: Relevance Maximization (RelMax) _How, and for which samples, does the model use the neuron in practice?_
- other work: cluster using t-sne and select from each cluster for diversity of samples
- zooming into reference sample by using receptive field information: heatmap how often pixels contributed

## Causal Interpretability for Machine Learning - Problems, Methods and Evaluation

- especially concerning fairness and bias: "if applicant had different protected features (e.g. race, gender), would outcome be different?"
- counterfactual analysis -> generate counterfactual explanations (both data and model counterfactuals)
- 4 categories:

1. CI and model-based interpretation: causal effect of model components on decision

- estimate ACE (average causal effect) of neuron on output
- model DNN as SCM [38] [12]
- have domain knowledge as causal graph
- GANs how and why images are generated, classes of concepts with dictionary [7]
- learning SCM during reinforcement learning of agents [68]

2. CI and example-based interpretation: generate counterfactual explanations

- minimal changes that change outcome
- suffers from Roshomon effect [71] -> multiple true explanations
- e.g. minimize MSE between models predictions and counterfactual outcomes
- or generate counterfactual examples
- reweigh distance with "median absolute deviation" -> more robust to outliers, sparser
- mask features that cannot be changed "counterfactually" e.g. age
- 2 criteria for counterfactual examples: 1 must be feasible in domain, 2 as diverse as possible
- use class prototypes for counterfactual explanations

3. CI and fairness:

- counterfactual: would same decision be made if protected feature changed?
- fairness measure: counterfactual direct, indirect and spurious effects
- formulate as constraint optimization problem [102]

4. CI as guarantee for interpretability

- transform any algorithm into _interpretable individual treatment effect estimation framework_ [92]
  Evaluation
- human subject-based metrics -> actual user studies
- others:
- does XAI method extract most important features
- faithfulness-> mask features
- consistency similar instances should have similar explanations
- counterfactual explanations should:
  - predict predifined output for cf expl.
  - have small amount of changes
  - be close to training data distribution
  - be (close-to) real time
  - be diverse
  -

## Discovering Causal Signals in Images

- observable foot-prints that reveal the “causal dispositions” of the object categories appearing in collections of images
- learning approach to observational causal discovery
- removing bridge from scene with car -> bridge causes (presence of) car
- dispositional semantics
- objects exercise some of their _causal dispositions_, or the _powers of objects_
- restrict to presence in scene
  **Hypothesis 1.** Image datasets carry an observable statistical signal revealing the asymmetric relationship between object categories that results from their causal dispositions
- use bounding boxes of objects, _object features_ activate mostly inside, _context features_ outside of bb
- distinguish causal (causes presence of object) vs anti-causal (is caused by object) features
  **Hypothesis 2.** There exists an observable statistical dependence between object features and anticausal features. The statistical dependence between context features and causal features is nonexistent or much weaker.
- we target causal relationships in scenes from a purely observational perspective
- similar to LinGAM?
- they generate artificial causal/anti-causal data to learn to distinguish this
- then test on real image data
-

## On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation

- for _Bag of Words_ models and _Neural Networks_
- explain _Pixel-wise Decomposition as a General Concept_
- taylor- or layer-based ?
- The important constraint specific to classification consists in finding the differential contribution relative to the state of maximal uncertainty with respect to classification which is then represented by the set of root points f(x0) = 0
- relevance needs to be positive or negative
- formula f(x) = (sum rel i+1) = (sum rel i) ...
- relevance of output layer: prediction itself
- relevance conservation property (the total relevance is constrained to be preserved from one layer to another, and the total node relevance must be the equal to the sum of all relevance messages incoming to this node and also equal to the sum of all relevance messages that are outgoing to the same node)
- more constraints, e.g. if node i has higher activation it should receive larger fraction of relevance score
- OR _first order Taylor approximation_:
**new read:**
- for bag of words methods and multilayered neural networks
- expect output of method to be >= 0


## Instance-wise Causal Feature Selection for Model Interpretation

- causal extension to instance-wise feature selection to explain black-box visual classifiers
- validate by measuring the post-hoc accuracy and Average Causal Effect of selected features on the model’s output
- a good instance-wise feature selection method should capture the most causal features in an instance
- -> most sparse and class discriminative features = most causal
- measure causal influence of input features with metric (that can be simplified to conditional mutual information sometimes)
- objective function for explainer using "continuous subset sampling"
- compare using post-hoc accuracy and special _average causal effect_
- explainer determines best explaining subset of features
- measure: "Relative Entropy Distance (RED)" -> non-linearity, markov property (only parents important)
- assumption: local influence-pixels are independent from each other !?
- causal strength = conditional mutual information of subset and output, given complement of subset
- maximal causal strength over subsets (further simplifying formula)
- need to approximate conditional distribution of P(Y|X\_-s) (output given complementing subset)
- -> use F(X\_-s)
- use function g: prob of being part of -s learned via a NN, then use gumbel-softmax continuous subset sampling

## Explaining Classifiers with Causal Concept Effect (CaCE)

- Many existing explainability methods rely solely on correlations and fail to account for confounding, which may result in potentially misleading explanations
- CaCE: causal effect of presence/absence of human-interpretable concept in deep neural nets prediction
- need to be able to simulate _do_-operator -> use VAE
- global explanation method (for whole class) -> difficult to mask out concept "male" -> VAEs
- conditional VAE
- general framework to quantitatively measure the causal effect of concept explanations on a deep model’s prediction
- conditional generative models to generate counterfactuals and approximate the causal effect of concept explanations
- claim that confounding does not exist for local explanation models
- for local explanations: mostly counterfactuals (pixel masking etc)
- Schölkopf: class label causes image (e.g. for digit 7),
- CaCE = ATE of do operator for concept E[f(I)| do(C_0 = 1)] - E[f(I)| do(C_0 = 0)] - or marginalized over all _a_ if C not binary
- **combat problem: network still learns real concepts, but XAI method shows confounded concepts**
- datasets: BARS (vertical/horizontal/green/red), colored-MNIST, CelebA, COCO-Miniplaces
- We find that a more complex classifier (ResNet-100) tends to be more affected by the correlation between class and color concept and results in higher CaCE values as compared to relatively simpler classifiers (such as simple-CNN).

## Unsupervised Causal Binary Concepts Discovery with VAE for Black-box Model Explanation

- concepts such as left/right/tob/middle/bottom stroke for letters
- _causal binary switches_ e.g. middle stroke OFF
- explain prediction with _X is Y because binary switch A is on and binary switch B is off_
- _concept specific variants_ e.g. different lengths of middle stroke (do not affect prediction)
- _global variants_ not tied to concepts, do not affect prediction, e.g. skewness
- they only use concepts that actually cause the output, e.g. not "sky" for "plane"
- implement user’s preference and prior knowledge as a regularizer to induce high interpretability of concepts
- no independece because "useful concepts for explanation often causally depend on the class information"
- use information flow to estimate causal effect -> equivalent to mutual information in proposed DAG
- interpretability regularizer: e.g. interpretable concept only affects small amount of input features, intervention of concept can only add/subtract pixel (not both), g = 1 -> pixels present, g = 0 not

## Towards Learning an Unbiased Classifier from Biased Data via Conditional Adversarial Debiasing

- main: novel adversarial debiasing strategy
- adversarial debiasing: have second loss that penalizes dependence between bias B and intermediate representation R
- main difference: replace B not _||_ R by B not _||_ R | L (L is label)
- 1st: choose criterion to determine whether classifier uses a (bias) feature
- 2nd: turn criterion into differentiable loss
- adversarial: have second NN that tries to find the bias
- 3 independence criteria that are continuous/differentiable: mutual information, Hilbert-Schmidt independence criterion, maximum correlation criterion
- for MI: use kernel density estimation, estimate individually on mini-batches
- for HSIC: some stuff with matrices (refers to [7] "kernel measures of conditional dependence")
- max-corr: partial correlation of classifier and bias-classifier: max PC(f(R), g(B) | L) = 0
- weirdly: completely dependent (no cross is violet, no rect green) for that test set
- works well for strongly but not completely biased training sets (or?)

## Salient Image Net: How to Discover Spurious Features in Deep Learning?

- use class activation maps to identify spurious features, test how much accuracy suffers when deactivating/masking those features
  **Steps:**

1. select neural features (scnd last layer) and visualize using highly activating examples
2. annotate neural features as "core" or "spurious"
3. extend human automation automatically to more samples
4. use mechanical turk again to validate
5. new dataset "Salient Imagenet" samples with core/spurious masks
6. assess accuracy by adding small noise to spurious/core visual attributes

- uses "feature attack" to visualize what the feature actually looks at (e.g. fence instead of cougar)
- previous work: select relevant features by selecting for highest mutual information between feature and model failure
- now: random subset of images from class, compute mean feature vector -> weight\*mean feature vector ~ contribution of features
- is basically the same as CRP?
- does not work for non-spatial spurious signals: race/gender

## Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)

- Concept Activation Vectors (CAVs): vector in direction of values (e.g. activations) of a concepts set of examples
- _E_m_ state of ML model, like input features and neural activations, _E_h_ human-interpretable concepts
- function _g: E_m -> E_h_ that is linear
- use examples to define concpets
- train linear classifier between concepts examples and random counterexamples, take vector orthogonal to the decision boundary
- goals: _accessibility, customization, plug-in readiness, global quantification_
- global explanation: explain whole class
- select set of images that show concept manually or chose other dataset where concept is labelled
- vector in space of activations of layer _l_
- CAV: normal to a hyperplane separating examples without the concept and examples with the concept
- pitfall: could learn meaningless CAV if chosen set of images is random(-ish)
- perform multiple training runs, meaningful concept should lead to consistent TCAV scores
- two-sided _t_-test, if null hypothesis TCAV=0.5 rejected, then it is related
- use _Relative CAVs_ e.g. between "brown hair" and "black hair" -> train linear classifier between two sets
- use activation maximization technique _empirical deep dream_ on CAVs to test how well they encode concept
- write noisy captions into images, noise parameter p \in [0,1] controls prop that caption agrees with image
- if network learned actual concept, it will perform well on images without caption
- evaluate if saliency maps communicate right thing to humans: humans not able to identify whether caption or image concept more important

## Network Dissection: Quantifying Interpretability of Deep Visual Representations

- qunatify interpretability of CNNs by evaluating alignment between inidividual hidden units and a set of semantic concepts
- "A disentangled representation aligns its variables with a meaningful factorization of the underlying problem structure"
- questions:
  1. what is disentangled representation, how quantify detect
  2. are interpretable hidden units special alignment of feature space
  3. what training conditions create better disentanglement
- emergent interpretability is "axis-aligned", can be destroyed by rotation, without affecting networks power
- match explanatory representations of CNN directly and automatically (using dataset)
- focus on quantifying the correspondence between a single latent variable and a visual concept
- To quantify the interpretability of a layer as a whole, we count the number of distinct visual concepts that are aligned with a unit in the layer
- activation map Ak (x) of every internal convolutional unit k is collected for each image
- to compare low-res to input image mask, scale up with bilinear interpolation
- binary mask, threshold activation by T_k (which is determined over whole dataset)
- compute intersections between activation masks and concepts masks -> intersection over union score IoU_k,c
- threshold how much they have to overlap at around IoU_k,c > 0.04
- one unit might be detector for multiple concepts, take top ranked label
- to quantify interpretability of layer, count unique concepts aligned with units _unique detectors_
- test how much humans agree with conv layers concepts
- measure axis-aligned interpretability: null hypo: "there is no distinction between individual high level units and random linear combinations of high level units"
- but other hypothesis true: change in basis (= rotation of representation space) does affect interpretability
- therefore: _interpretability is neither an inevitable result of discriminative power, nor is it a prerequisite to discriminative power_
- batch normalization decreases interpretability significantly, drop out reduces object detectors in favor of textures
- benchmark deep features of trained models on a new task by taking representation at last layer and train linear svm

## Understanding the Role of Individual Units in a Deep Neural Network

- we wish to understand if it is a spurious correlation, or if the unit has a causal role that reveals how the network models its higher-level notions about trees
- not _where_ network looks (saliency maps) but _what_ it is looking for and _why_
- tasks image classification and image generation
- test the _causal_ structure of network behavior by activating and deactivating the units during processing
- remove e.g. 20 most important units

## Ground Truth Evaluation of Neural Network Explanations with CLEVR-XAI

- compare 10 different XAI methods based on CLEVR-XAI dataset
- methods compared: Class Saliency Map, Grad-CAM, Gradient x Input, Integrated Gradients, LRP, Excitation Backpropagation, Guided Backpropagation, SmoothGrad, VarGrad
- previously validated using pixel perturbation analyses, or using pixel relevances as object detection signal (make bounding box)
- for question answering task not every object important for explanation -> better to evaluate
- novel quantitative metrics for heatmap evaluation: relevance mass accuracy and relevance rank accuracy
- old evaluation (pixel perturbation) with 2 schemes: random pixel flipping, brute force search
- problem with perturbation: images lie outside of data distribution
- other approach: train multiple times with randomly permuted class labels, measure similarity
- pointing game accuracy: singlest most relevant pixel within bounding box of object
- problem with IoU (intersection over union): classifier might only need part of object, but this favors fully covering object

## Explaining Visual Models by Causal Attribution

- based on Causal Counterfactuals -> for generative models
- contrasts with salience<: want to know which _latent factors_ e.g. hair color, use of make-up influence outcome: more semantically charged
- answer "given face of woman, classified as woman, no beard: what is prediction if there had been a beard"
- If we notice that by intervening on a factor the prediction changes significantly, we can say that the current value for that factor is a cause for the current prediction
  contributions:

1. method for implementing causal graphs with Deep Learning: Distributional Causal Graphs
2. new explanation technique: Counterfactual Image Generator
3. limitations

- use log-likelihood for any latent factor sample
- "Distributional Causal Graph" (DCG)
- experimantal dataset: use "fake" class as causal generator: type influences known generating factors

## Generative causal explanations of black-box classifiers

- based on learned low-dimensional representaton of data
- use generative model and information theoretic measures of causal influence
- does not require labeled attributes or knowledge of causal structure
- challenge: changing factor only meaningful if changed occurs naturally
- SCM: relates independent latent factors, classifier input, classifier output
- use independence of latent factors to show that causal influence of latent factors cna be quantified with mutual information
- optimization problem for learning mapping from latent factors to data space
- is a "glocal" method: learns global, to show local
- not one-to-one correspondence between indvidual latent factors and semantically meaningful features, but separate latent factors relevant for prediction
- DAG should: describe functional causal structure of data, explanation from output Y not ground truth, DAG has link X->Y
- changing causal factors: if causal to classifier, class changes, otherwies only style (MNIST)
- visualization: small multiples of generated images while changin factor (either class changes or stays the same)
  [4] [13] [47]

## CASTLE: Regularization via Auxiliary Causal Graph Discovery

- regularization improves generalization of supervised models to out-of-sample data
- causal direction has better accuracy than anti-causal
- CASTLE: causal strucure learning regulaization
- learns DAG as adjacency matrix embedded in nn input layers
- common regularization techniques: data augmentation, dropout, adversarial training, label smoothing, layer-wise strategies
- other: supervised reconstruction: hidden bottleneck layers to reconstruct input features
- reconstruct ony input features that have neighbours in causal graph

## Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps

- showing heatmaps (and cutting according to segmentation)
- showing strongest activating image (deep dreamed or something) for different classes

## Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces

- this is the paper with PCA?
- disentangle explanations by finding relevant subspaces in activation space that can be mapped to more abstract human-understandable concepts and enable a joint attribution on concepts and input features
- extend PCA to PRCA (principal relevant component analysis) and disentangled relevant subspace analysis (DRSA)
- optimizing relevance when reducing dimensionality?
- works alongside common attribution techniques: Shaple Value, Integrated Gradients, LRP
- need "richer structured explanations" -> "higher-order explanations"
- CRP mentioned as a "hierarchical explanation"
- 3 use cases: detecting and removing Clever Hans, understanding relation between similar classes, show manipulation of adversarial perturbations of input
- assume mapping I -> K (k in K are sub-strategies/neurons/concepts) -> Y
- uses same idea as CRP: filtering on concept k in backpropagation
- build linear model from activations to inferred latent concepts, use like encode decoder but with exact reconstruction -> U^T U = I_d' (U in R^(d\*d'))
- one solution: d' eigenvectors associated with highest eigenvalues of symmetrized cross-covariance matrix
  E[ac^T + ca^T]
- takes into account the response to activation a via context vector c
- therefore PRCA ignores high variance directions if model responds invariant or negatively to these variations
- DRSA: relevant and _disentangled_ e.g low spatial overlap

## Inducing Causal Structure for Interpretable Neural Networks

- learning from data while inducing a deterministic program/bayesian network (=SCM)
- by aligning neural model with causal model
- _interchange intervention training_ IIT
- model has counterfactual behavior of causal model
- "[...causal... ] insights have the potential to make up for gaps in available data, or more generally to provide useful inductive biases"
- interchange interventions: swapping of internal states created for different inputs
- seems kinda ridiculous to try and enforce a causal structure for a NN

## Neural Network Attributions: A Causal Perspective

- nn is viewed as SCM, computing causal effect from that
  -this approach induces a setting where input neurons are not causally related to each other, but can be jointly caused by a latent confounder
- We note that our work is different from a related subfield of structure learning, where the goal is to discern the causal structure in given data. The objective of our work is to identify the causal influence of an input on a learned function’s (neural network’s) output
- non-identifiability of “source of error”: _"It is impossible to distinguish whether an erroneous heatmap is an artifact of the attribution method or a consequence of poor representations learnt by the network"_
- do not explicitly attempt to find the causal direction
- for Recurrent Neural Networks: need to "time unfold" them
- compute interventional expectations (changing other inputs while keeping one fixed)
- setting: assume independency of input neurons
- compute ACE (average causal effect) using interventional expectation and baseline
- -> define baseline: E_xi[E_y[y | do(xi = a)]]. in practice: perturb input neuron xi uniformly in fixed intervals [low_i, high_i] and compute interventional expectation
- interventional expectation: is function of xi, assume polynomial
- use "Bayesian model selection" to determine optimal order of polynomial by maximizing marginal likelihood given data
- learn _causal regressors_ and obtain ACE by evaluating them at xi = a and subtract from baseline_xi
- for recurrent nns: can't use distribution of data, estimate means and covariances after evaluating RNN for each input sequence with value at xi = a
- would usually need Hessian for calculating interv. exp. -> estimate using taylor decomposition

## Towards Higher-Order & Disentangled XAI (Montavon)

- for scientific applications -> find influential proteins in networks
- predict proteins from other proteins with best possible accuracy
- multiplication example: f(x) = x1*x2 + x3 -> 6 = 3*1+2 or 6 = 2\*2 + 2?
- better: "higher-order explanation": x1 and x2 contribute jointly

## Theoretical Behavior of XAI Methods in the Presence of Suppressor Variables

- simple 2D artifical dataset that is capable of creating "supressor" variables (vairables independent of prediction target)
- suppresors variable -> like watermark: w->X, Y->X (X=target is a collider -> selection bias)
- a feature is important if it has a statistical dependence with target variable
- experiment: X1 = z+e where z=y, X2 = e, -> perfect prediction of z = X1- X2 = z +e -e = y
- watermarks are confounders? (according to Sebastian Lapuschkin)
- "in practice, XAI methods do not distinguish whether a feature is a confounder or a suppressor, which can lead to misunderstandings about a model’s performance and interpretation"
- Linear Generative Model: x= az+e, y=z, z = Rademacher(1/2), a=(1,0)^T,, H = N(0,SIG) where SIG is covariance matrix
- argument: for linear models suppressor variable should be irrelevant, it is "glass box". however XAI methods attribute non-zero importance
  -> even a "bayes optimal classifier" attributes importance -> need to know suppressor variable to accurately predict
- Haufe: covariance between model output and each input feature -> global importance map: "linear activation pattern" -> works
- faithfulness + pixel flipping assigns importance to suprresor feature:
- simplest form: set corresponding weight of feature to 0 -> squared error of expectation of accuracy with and without feature -> does not work
- permutation: random permutations of feature -> also assigne importance, does not work
- partial dependency plots: does not work with correlated data
- **marginal plots**: vis. assess scatter plots of y as function of individual features -> vainishing importance of suppressor var -> works???
- shapley values: difference between "value functions" of all subsets including feature and all subsets not including feature
- assess differenct value functions: "coefficient of multiple determination"
- value function "R² measure", shapley values work because x2 has no importance assigned
- SHAP: "shapley additive explanations" (approximate by assuming feature independence), can be extended with conditional expectation
- counterfactual explanations: for linear: just closest point of decision boundary -> includes shift in x2 -> does not work
- FIRM (feature importance ranking measure) -> conditional expectation ~ do-operator? -> works
- integrated gradients: does not work
- LIME: constructs "glass box" linear model in local neighbourhood -> real linear model assigns weight to x2 -> does not work
- Saliency Maps, LRP, Deep Taylor decomposition (DTD), DTD works sometimes, depends on parameters, "DTD can generally yield almost any explanation"
- methods that "work" use the statistics of the training data including correlation structure of data and not just model
- whole idea is questionable: if "suppressor variable" is necessary to predict accurately, it should have importance!
- most methods only fail when correlation is present between x1 and x2 (watermark and shape)

## Restricting the Flow: Information Bottlenecks for Attribution

- first impression: similar to CRP but adding noise to intermediate feature masks (instead of boolean masks?)
- information-theoretic: estimate the information a pixel has
- "information bottleneck method": "bottleneck" inserted into neural networw, restricts flow by adding noise to activation maps
- unimportant actiations replaced by noise -> learn parameters per sample or whole dataset ("readout")
- IBA (information bottlenec attribution)-> "theoretic upper-bound on used information, demonstrating strong empirical performance"
- "loss function" minimize information flow (optimize intensity of noise) and maximize classification accuracy
- function: introduce new random variable Z, maximizes shared info with Y, minimize with X: max(I[Y;Z] - b \* I[X;Z])
- insert bottleneck "where nn still has local information" e.g. after conv layers
- R is intermediate representation at this layer, add noise to it with factor lambda for each feature (lambda*R + (1-lambda)* noise)
- assume independence of features (!) - dumb

## When Explanations Lie: Why Many Modified BP Attributions Fail

- analyze lots of backpropagation XAI methods: DTD, LRP, ExcitationBP, PatternAttribution, DeepLIFT, Deconv, RectGrad, GuidedBP
- they find that explanations independent of parameters of later layers except for DeepLift
- new evaluation metric: "CSC" - cosine similarity convergence
- for CuidedBP last layer is fully irrelevant -> problem!
- modified BP: backpropagate custom relevance score instead of gradient
- z+ rule (used by DTD, LRP, ExcitationBP) yields multiplication chain of non-negative matrices
- Our findings show that many modified BP methods are prone to class-insensitive explanations and provide saliency maps that rather highlight low-level features
- Negative relevance scores are crucial to avoid the convergence to a rank-1 matrix — a possible future research direction
- z+ rule: only bp positive values
- theoretical problem: all those methods converge to a rank 1 matrix
- why not noticed: often only single class prediction tasks (class insensitivity is irrelevant)

## A Rigorous Study Of The Deep Taylor Decomposition

- fundamental assumption of taylor theorem: DTD roots lie in same linear regions as the input -> not true in empirical evaluation
- many methods simplify to make accessible to human eye: local neighbourhood, assume linearity (gradient-based) or independence (shapley approximation)
  summary:

1. proof that root points must be contained in same linear region as input
2. LRP0: if root points locally constant, relevances are similar to input x gradient
3. DTD underconstrained: if root points depend on lyers input, DTD can create arbitrary explanations
4. DTD cannot extend to analytic activation functions (e.g. Softplus)
5. train-free DTD does not enforce root points locations
6. validate empirically
7. reproducibility study of Clever-XAI paper -> DTD is black-box, hard to evaluate explanation quality empirically

- problem shown in clever-xai images: it will only show positive evidence -> last layer is irrelevant. correctly identifies important region, but has class insensitivity -> if other classes prediction output is > 0 it will give same explanation
- reproduced clever-xai test, "then compared 1000 saliency maps for the correct answer, an incorrect answer (but from the same category), and the correct answer but a different question"
- they scaled saliency maps (maybe that is an issue?)
- indicates that "information leakage" between questionand LRPs saliency map is present

## Sparse Subspace Clustering for Concept Discovery (SSCCD)

- motivation: reveal coherent, discriminative structures exploited by model, rather than accordance with human-identified concepts
- concepts not one- but low-dimensional subspaces
- approach does not require reinforcing "interpretability" e.g. enforcing dissimilarity of concepts
- concept interpretability method: a. concept discovery, b. mapping feature - input space c. relevance quantification
- for b. translate feature level concept maps to the input level by simple bilinear interpolation
-

## Identifying Interpretable Subspaces in Image Representations

- method named: Automatic Feature Explanation using Contrasting Concepts (FALCON)
- captions its highly activating cropped images using a large captioning dataset (like LAION-400m) and a pre-trained vision-language model like CLIP
- FALCON also applies contrastive interpretation using lowly activating (counterfactual) images, to eliminate spurious concepts
- We show that features in larger spaces become more interpretable when studied in groups
- natural language explanations as a complement to heatmaps and visual explanations
- remove words/ captions that are not highly activating given neuron

## But that's not why: Inference adjustment by interactive prototype deselection

- concepts learned by NN can neither be accessed nor interpreted directly
- interaction in segmentation tasks has clear focus on "where" aspect, "what" aspect is also important, but harder to access
- evidence from cognitive psychology suggests that human
- cognition relies on conceptual spaces that are organized by prototypes
- "prototypical part networks"
- propose new method: "deep interactive prototype adjustment (DIPA)"
- Prototype-based Learning: Prototype-based deep learning builds on the insight that similar features form compact clusters in latent space
- Prototypes are understood as concepts that are more central for a class as compared to instances which reside close to the decision boundary

## Explaining nonlinear classification decisions with deep Taylor decomposition (DTD)

- 2 properties should be satisified by heatmap/relevance scoring:
  1. _conservative_ the sum of assigned relevances in the pixel space corresponds to the total relevance detected by the model
  2. _positive_ all values forming the heatmap are >= zero
     together _consistency_
- consistent heatmap is 0 everywhere if f(x) = 0 -> empty heatmap if no object detected
- **natural decomposition**: prediction function: f(x) = sum_p sigma_p(x_p) where sigma are set of positive nonlinear function applying to each pixel. If there is deactivated state x0_p such that sigma_p(x0_p) = 0, the R_p(x) is effect on prediction of deactivating pixel p
- **taylor decomposition**: taylor expansion of function at some well chosen root point **x0** where f(x0) = 0
- _first order taylor expansion_: f(x0) + gradient at x _ (x-x0) + e = 0 + sum over pixel of gradient of f at pixel _ (x_p - x0_p) + e
- in words: heatmap is element-wise product between gradient of function at root point
- good root point: removes what causes positive f(x) but minimally changes image
- if x, x0 in \R then gradient points to same direction as x-x0
- obtain root by minimizing L2 of z with f(z) = 0 to x min_z||z-x||² -> expensive, not generally solvable due to convexity
- practically: nearest point x0 often not visually different
- hard to find nearest root in constrained input space -> further restrict search domain to subset of X
- for ReLU: has to be positive -> z_ij+ = x_i\*w_ij+ only positive weights
- _relevance model_: maps set of neuron activations at given layer to relevance of a neuron in a higher layer
- output of relevance model can be redistributed onto its input variables to backpropagate
- min-max relevance model:
  - y_j = max(0, sum_i(x_i\*v_ij + a_j) ) and R_k = sum_j(y_j)
  - a_j = min(0, sum_l(R_l\*v_lj + d_j) ) -> negative bias=inihibitor: prevents relevance activation of no upper-layer neuron uses it
- training-free relevance model: assume something as constant

## Learning how to explain neural networks: PatternNet and PatternAttribution

- other methods fail on linear model -> should "work reliably in the limit of simplicity, the linea models"
- test methods on linear generative data and linear model
- uses same example as _"Theoretical Behavior of XAI Methods in the Presence of Suppressor Variables"_ with supressor/distractor variable
- "PatternAttribution is a DTD extension that learns from data how to set the root point"

## Axiomatic Attribution for Deep Network

- paper about Integrated Gradients -> averaging the gradient through multiple attribution steps

## The (Un)reliability of Saliency Methods

- In order to guarantee reliability, we posit that methods should fulfill input invariance
- "implementation invariance": functionally equivalent networks (different architecture, same output for all inputs) always attribute in an identical way
- "input invariance":t a saliency method mirrors the sensitivity of the model with respect to transformations of the input
- evaluated by comparing the saliency heatmaps for the predictions of network 1 and 2, where xi2 is simply the mean shifted input (xi1 + m2 )
- A saliency method that satisfies input invariance will produce identical saliency heatmaps for Network 1 and 2 despite the constant shift in input
- bold assumption: network 1 and network 2 have "identical f(x)"
- gradient and signal methods (guided Backprop, DeConvNet..) are input invariant -> attribution entirely function of weights
- gradientXinput fails, as input shift is carried thorugh to final attribution
- IntegratedGradients and DTD are dependent on "reference point" -> "baseline" for IG and "root point" for DTD
- reference point is hyperparameter of those methods
- input invariance (e.g. adding checkered pattern to image) fails for all methods except DTD with "PA" root point
- PA = PatternAttribution achieves input invariance
- for LRP root point = zero vector
- for PA root point = natural direction of variation in the data -> determined by covariance of data, compensates for input shift
  PA root point: x0 = x - a*w^T*x, where a^T*w = 1 -> in linear model a = cov/weights*cov
- reference point is important!

## From Clustering to Cluster Explanations via Neural Networks

- using LRP style explanation method to attribute clustering decision to input features
- turning clustering method into "neural network" by "neuralizing" it

## Coherence Evaluation of Visual Concepts With Objects and Language

- "how to automatically assess meaningfulness of unsupervised visual concepts using _objects and language_ as forms of supervision"
- propose Semantic-level, Object and Language-Guided Coherence Evaluation (SOLaCE)
- assigns "semantic meanings" in the form of words to concepts and evaluates "degree of meaningfulness"
- with user study that confirms that evaluations highly agree with human perception of coherence
- thesis: "_meaningful visual concepts have concise descriptions in natural language_"

## Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations

- "The key idea behind the unsupervised learning of disentangled representations is that real-world data is generated by a few explanatory factors of variation which can be recovered by unsupervised learning algorithms"
- "theoretically show that the unsupervised learning of disentangled representations is fundamentally impossible without inductive biases on both the models and the data"
- often assumed: 2-step generative process: random latent Z (variance/"meaningful" noise), X sampled from P(X|Z) -> basically f(x) is extracting useful info
- claim of other work: _disentangled_ representations should be able to "integrate out nuisance factors, perform interventions and answer counterfactual questions (pearl)"
- SOTA approaches of _unsupervised disentanglement learning_ mostly **Variational Autoencoders**
- investigate current approaches with large scale experiment (12000 models)
- challenged common beliefs:
  - all methods effective at decorrelating posterior, but dimensions of representation are correlated
  - random seeds and hyperparameters matter more than model choice -> not "unsupervised"
  - cannot validate assumption that disentanglement useful for downstream tasks
- future research therefore: research on inductive bias, why do we need disentanglement, more reproducible experiments, test on datasets of varying complexity

## Explaining Deep Learning Models using Causal Inference

- general framework idea to interpret NN as SCM
- ability to perform arbitrary causal interventions
- not like "ablation testing" (turning features on and off)
- this approach only one time construction of causal abstraction
- use inherent DAG structure of NN model as skeleton for causal model
- need to select appropriate transformation function to answer the right _what if_ questions
- from convolution response matrix to real number phi: R^(p\*q) -> R
- learn function r = f(PA_r) (r is function of its parents)
- function for j-th filter in i-th layer can be approximated with r_ij = f_ij(R^i-1)
- simplest transformation phi: binary: filter has high variance or not
- other phi: take frobenius norm of matrix
- sanity check: using SCM as prediction model (using frobenius norm transformation)
- **_might be good as a reference point to start from_**

## Causal Learning and Explanation of Deep Neural Networks via Autoencoded Activations

- _LOL: In highly sensitive domains like credit scoring or **money laundering**_
- contributions:
  - causal model of DNN formulated with human-understandable concepts
  - unsupervised extraction of concepts highly likely to be human understandable
  - measure causal effects of inputs and concepts on DNNs outputs
- formulation of explanation as "What changes can be made to the input for the output to change or stay the same?"
- "For example, gradient-based methods of explanation such as layerwise relevance propagation (LRP) and Grad–CAM attempt to explain the output activation of a class C in terms of the input P activations, realized for a specific input Pj . While these methods are certainly useful, they don’t provide the causal intervention semantics that are sufficient for robust explanation."
- adversarial examples show that local behavior of DNNs "does not have semantic relevance"
- in this work compute _expected causal effect_ (compare effect of intervention to no intervention)
- if model predicts correctly, then causal effect of any variable on false output is zero (if output binary)
- which concepts to use? one possible way: activations, but "specific representation of instance features given by activation values does not necessarily have any special relevance"
- concept representation = transformation on activations
- desirable properties of concept representation:
  1. low-dimensional
  2. concepts are interpretably ("contiguous areas containing consistent interpretable visual features")
  3. should contain all relevant info needed for achieving target networks task
- uses autoencoder to construct concepts
- loss function of autoencoder is "shallow": L1 norm of difference between input and output activations
- but also "deep" loss incorporating total accuracy of network
- series of autoencoders for each layer of activations

## Conditional dependence tests reveal the usage of ABCD rule features and bias variables in automatic skin lesion classification

- ABCD rule: Asymmetry, Border, Color, Dermoscopic structures
- thresholding score out of ABCD features yields high accuracy to detect melanoma
- features such as asymmetry do not correspond to an image region
- not relevant because "feature of interest" is predetermined in this case
- also the method treats model as black box (model agnostic)

## Causes of Outcome Learning: a causal inference-inspired machine learning approach to disentangling common combinations of potential causes of a health outcome

- not just one but multiple/combinations of causes:
- seeks to discover combinations of exposures that lead to an increased risk of a specific outcome in parts of the population
- define causal model, fit model on additive scale, decompoes ris contributions, cluster individuals based on risk contributions, hypothesis development
- study synergistic effects A + B > A and B individually
-

## Understanding Failures of Deep Networks via Robust Feature Extraction

- method called "Barlow" inspired from _Fault Tree Analysis_
- approach can be used to reveal _spurious correlations_ and _overemphasized features_
- Ideally, we would like to find clusters that jointly have large error rates and that cover a significant portion of the total errors from the benchmark
- most activating images -> not clear which part of image
- heatmaps -> not clear if overlapping
- feature attack
- use top6 max activation images, corresponding heat maps and feature attack images for desired feature

## CXPlain: Causal Explanations for Model Interpretation under Uncertainty

- "we frame the task of providing explanations for the decisions of machine-learning models as a causal learning task, and train causal explanation (CXPlain) models that learn to estimate to what degree certain inputs cause outputs in another machine-learning model"
- conrtibutions:
  - causal explanation models (CXPlain) -> train causal model on explaining any ML model
  - use "bootstrap resampling" to derive uncertainty estimates for feature importance score
  - experiments show: fast and more accurate than other model-agnostic methods
- grangers definition of causality: better able to predict Y if X present than only all other variables
- assumption: causal sufficiency (all relevant variables known), x temporally before y
- "without input feature" -> mask with zero??? or replacing with mean value or considering (cond?) distribution of feature
- use loss function to compute predictive error -> causal effect = degree of reduction in error
- **causal objective**: _L_causal = 1/N sum_l KL(Omega_X_l, A_X_l)_ -> minimize KL-divergence between target importance distribution Omega for given sample X and distribution of importance scores A
- train supervised learning models on causal "loss" _L_causal_
- for high-dimensional images -> group pixels

## Pruning by Explaining: A Novel Criterion for Deep Neural Network Pruning

- for transfer-learning setups
- good for "resource-constrained" application -> little data, no fine-tuning
- scenario 1: prune pre-trained models, afterwards fine-tuning
- scenario 2: transfer to data-scarece new task with constrained time, computational power/energy -> mobile/embedded applications
- (for myself/ causal cause: if you can "prune" certain neurons, their causal effect must be none or extremely small)
- outperforms other methods quite significantly in their experiments (except in one case "weight magnitude pruning")

## Revealing Hidden Context Bias in Segmentation and Object Detection Through Concept-Specific Explanations

- for segmentation: "heatmap" just resembles segmentation itself (just tresholded)
- glocal/ CRP: _where_ the model pays attention to and _what_ it sees
- "Especially in the multi-object context, it is crucial to attain object-class-specific explanations, which is not possible by the analysis of latent activations: Concepts with the highest activation values can refer to any class that is present in the image, such as the horse’s rider, or none at all, since activations do not indicate whether a feature is actually used for inference."
- basically same content as main paper, showing how to use concept relevance maximization to visualize features and find biases/context dependency -> "context bias"

## From Hope to Safety: Unlearning Biases of Deep Models by Enforcing the Right Reasons in Latent Space

- using CRP to identify biases und then "unlearning" -> dampening effect of bias neurons/activations
- effectiveness of "Class Artifact Compensation (ClArC)" using Concept Activation Vectors (CAVs) limited by only targeting activations -> not class specific, may only partially unlearn due to "methods indirect regularization"
- new method "Right Reason ClArC": _explicitly penalizes latent gradient along bias direction_
- only requires sparse sample-wise label annotation
- annotations can be acquired semi-automatically using XAI tools
- "post-hoc model correction based on only few fine-tuning steps"
- paper compares mostly to TCAV
- Augmentive ClArC: adds artifact to all samples to make model invariant in that direction
- Projective ClArC: suppress artifact direction during test phase, does not require fine-tuning
- bias concept activation vector (CAV) **h** loss term explicitly penalizes magnitude of this vector with L_RR = (feature gradient \* h)^2
- to ensure that bias direction **h** stays constant: freeze weights of layers <= l during fine-tuning phase
- ablation study: images that should use "bias" feature, should not suffer in accuracy (e.g. digital clock uses date stamp feature)
- only uses one layer (last convolutional layer in this case) to find bias concepts

## Reveal to Revise: An Explainable AI Life Cycle for Iterative Bias Correction of Deep Models

- framework combining lots of methods from frauenhofer people in one
- life cycle:
  1. Identification of model weakness
  1. a) explanation embedding: cluster images, find outliers, visualize heatmaps of outliers using **SpRAy**
  1. b) Concept embedding: **CRP** concept visualization
  1. Artifact labeling & localization
  1. a) finding artifact direction (CAV) (e.g. through clustering)
  1. b) localizing artifact (heatmaps)
  1. Model Correction -> unlearn artifacts (using RR-ClArC?)
  1. Model Evaluation
  1. a) poisoning dataset
  1. b) Artifact Relevance

## ContrXT: Generating contrastive explanations from any text classifier

- how does model change prediction in contrast to previous models (e.g. due to retraining)
- not relevant

## Persuasive Contrastive Explanations for Bayesian Networks

- combine questions "why correct" and "how not correct" (to "why outcome t instead of t' ? ")
- they identify 4 explanation methods:
  1. explanation of evidence
  2. explanation of reasoning
  3. explanation of the model itself
  4. explanation of decisions
- paper not really relevant: focuses on bayesian networks

## Towards Best Practice in Explaining Neural Network Decisions with LRP

- used to apply same rule to all layers, now specific rules:
- COMPOSITE rule = LRP_CMP->
  - fully connected layers: LRP_epsilonwith e << 1
  - conv layers: LRP_alpha_beta with alpha,beta in {1,2}
  - input layer LRP_b /DTD_Z_B

## Learning to Explain: An Information-Theoretic Perspective on Model Interpretation

- instance-wise feature selection: select information-theoretically closest features for each sample
- maximize mutual information between selected subset of features and response variable
- firect estimation of MI and discrete subset samping are intractable -> apply lower bound for MI an develop continous reparametrization of sampling distribution
- results dont seem to be very good

## On the Relationship Between Explanation and Prediction: A Causal View

- look at treatment effect of prediction on explanation when blocking paths to hyperparameters
- effect is small (probably because hyperparameters and input data make all the difference)

## XAI-TRIS: Non-linear benchmarks to quantify ML explanation performance (Benedict Clark, Rick Wilming, Stefan Haufe)

- suppressor variables in non-linear benchmarks
- suppressor variable e.g.: background lighting helps with normalizing objects colors
- 1 linear, 3 non-linear binary image classification problems
- 2 suitable metrics based on signal detection theory and optimal transport
- use different kinds of background noise to study effec of suppressor variables on explanation performance
- evaluate 16 XAI methods, 3 machine learning architectures
- performance metric:
  - earth mover distance (EMD) / Wasserstein metric
  - importance of pixels in feature (tetris shape in this paper)
  - cost of transforming importance map output into mask of this feature
  - using euclidean distance between pixels
  - normalize EMD score

## Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset

- compare "baseline" (just predictions ordered by confidence) with some other XAI methods
- other methods:
  - "Invertible Neural Networks": moving along weight vector changes prediction
  - "Concepts (Zhang)": automatically-discovered concepts (NMF approach), non-negative matrix factorization -> non-negative TCAVs
- claim: first to extensively test counterfactual-based and concept-based explanations on bias discovery using a challenging dataset
- Spatially overlapping attributes, like color and shape, directly challenge saliency map explanations.
- might be good measure for ground-truth feature importance: "PREDICTION FLIP" measure difference in prediction with feature on/off
- calculated a linear fit for each parameter change to the logit change. We reported the coefficient of determination R2 , which indicates how much of the variance in the prediction can be explained linearly by the analyzed property

## From attribution maps to human-understandable explanations through Concept Relevance Propagation

- for pixel map might be clear where important information can be found, but not what characteristics of the raw input features the model has extracted and used during inference, _or whether this information is a singular characteristic or an overlapping plurality thereof_
- they do user study with RelMax, against other attribution map-based
- determining the flow of relevance via controlled masking operations in the backwards process -> what are controlled masking operations?
- condition sets theta are configured automatically: ranking units in descending order of relevance,
- advantage of RelMax: conditional maximization: how does model use latent feature for different classes, not just in general
- relate quite a lot to negative relevance?
- for their user study -> easier to identify clever hans than to reject existence
- clarity suffers from CRP -> too complex, users prefer simple explanations
- for my experiment: would be cool/important to know whether CRP identifies if artifact is relevant vs not
- method for comparing filters: averaged cosine similarity on reference samples

## When are Post-hoc Conceptual Explanations Identifiable?

- concept discovery should be identifiable, meaning that a number of known concepts can be provably recovered to guarantee reliability of the explanations
- use automatic/ unsupervised concept discovery
- unsupervised concept spaces can be highly distorted
- methods such as PCA and ICA (independent component analysis) cover independent non-gaussian ground truth components. (no causal/spurious links)
- in practice: complex dependencies, generative models often use gaussian distribution
- concept discovery method based on _independent mechanisms_
- identifiability is important concept for my thesis. Statement: "if a known number of _ground truth components_ generated the data, the concept discovery method provably yields concepts that correspond to the individual ground truth components and can correctly represent an input in the concept space"
- utilize "_visual compositionality properties_": tiny changes in (generative) components affect input images in orthogonal or even disjoint ways
- -> new discovery method from this using "disjoint/independent mechanisms" criterion
- **disjoint mechanism analysis (DMA)**:
- **independent mechanism analysis (IM A)**:
- Träuble et al. (2021) shows that even if just two components of a dataset are correlated, current disentanglement learning methods fail
- unsupervised disentanglement, without further conditions, is impossible (Hyvärinen and Pajunen, 1999; Locatello et al., 2019; Moran et al., 2022)
- using idea of concept activation vectors (CAVs)
- **terminology**: _components_ in ground truth, _concepts_ in learned representations/directions
- faithful encoder: ground truth components are recoverable with full rank, f is lazy and invariant to changes in x which cannot be explained by ground truth components
- -> sufficient to find encoder Mf whose Jacobian MJ_f has disjoint rows
- searching for an M Jf with orthogonal (instead of disjoint) rows permits post-hoc discovery of concepts. We refer to is property of M Jf as the _IMA criterion_.
- "identifiable" necessary for our work

## Invertible Concept-based Explanations for CNN Models with Non-negative Concept Activation Vectors

- Non-negative concept activation vectors *perform best in computational and human subject experiments*
- saliency maps *only point out important area* and dont *identify key concepts*
- ACE (automated concept based explanations): learn clustering of image segments from CAVs
- ACE drawbacks: learned concept weights inconsistent for different instances, hard to measure performance,
- information can be lost: in unused segments, distance between segments, cluster centroids...
- NCAV sort of similar to ACE, cause k-means ~ dimensionality reduction ~ matrix factorization
- in addition to DR, matrix factorization also analyzes information loss with inverse function
- through inverse function of matrix factorization, information lost in in explanation can be measured
- NMF (non-negative matrix factorization): gives global but also local explanations through inverse function
- **interpretabiliy** and **fidelity**
- main idea: reduce feature space of neurons dimensionality
- measure for **fidelity**: 0-1 loss -> accuracy loss when predicting from F' instead of F
- 

## CRAFT: Concept Recursive Activation FacTorization for Explainability
- "Where" vs "What" question
- recursive strategy to detect and decompose concepts across layers
- faithful estimation of concept importance using Sobol indices
- use implicit differentiation to unlock Concept Attribution Maps
- CRAFT uses non-negative matrix factorization

## Multi-dimensional concept discovery (MCD): A unifying framework with completeness guarantees
- *completeness axiom:* attributions sum up to the model prediction
- Concepts as multi-dimensional subspaces: space spanned by hidden neurons is decomposed in the most general way:
- not just allow rotation and non-orthogonality but also multi-dimensionality
- allow rotation = Zhang 2021, PCA
- allow non-orthogonality = concepts can be linearly independent but not orthogonal
- allow multi-dimensionality = concept can lie on hyperplane spanned by multiple neurons
- *most faithful* capture any meaningful structure within hidden feature layer
- covers relevant feature space with fewer concepts, reaches specified level of completeness earlier
- method: 
  (i)  randomly choose and cluster a set of feature vectors with any clustering algorithm
  (ii) construct subspace bases for all clusters Cl via PCA
- use sparse subspace clustering (SSC) for clustering of feature maps
- test concepts by masking out images with a classical inputation algorithm (Bertalmia 2001)
- MCD-SSC is actually not the best, but they argue that the ones that are better consistently only find one concept and therefore have no benefit over classical attribution methods like LRP

## iNNvestigate neural networks! (Alber2018)
- code and reference implementation for many XAI methods
- not important

## Benchmarking Attribution Methods with Relative Feature Importance (Been Kim, Mengjiao Yang) (Yang2019)
- **relative feature importance** similar to wm/(wm+shape) 
- a priori knowledge of relative feature importance
- 1. carefully crafted dataset with known feature importance
- 2. 3 metrics to quantitatively evaluate attribution methods
- certain methods tend to produce *false-positive* (not actually important marked important)
- *relative*: we do not know *absolute* importance of a feature, but can identify *relative* feature importance, compared to another model -> by changing frequency of features in dataset
- metrics compare attributions between pairs of models *model dependence* and pairs of input *input (in)dependence*
- train one model on object and one on scene (background) of images
- take average attribution that saliency method applies to image region (object or background)
- *CF* common feature (or bias) -> with different degree
Metrics:
1. Model Contrast Score (MCS): difference between concept attributions between the two models
2. Input dependence rate (IDR): percentage of correctly classified inputs where CF is attributed less than original regions covered by CF
3. Input Independence rate (IIR):
