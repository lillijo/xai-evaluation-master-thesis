# Papers important for related work and theoretical background:

## 1. XAI in General and Important Methods:

### Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications (Samek2021)
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
2. Taylor decomposition: approximating function through first few terms of taylor series (derivatives)
3. Deep Taylor decomposition: perform taylor decomposition at each layer of dnn
   **Analysis of large number of heatmaps:**
- first goal: find data artifacts - clever hans effects
- second goal: investigate learning model to find novel prediction strategies

### Unmasking Clever Hans predictors and assessing what machines really learn (Lapuschkin2019)
- **SPRAY**
- standard performance evaluation metrics bad at distinguishing short-sighted and strategic problem solving behaviors
- use SPectral Relevance Analysis
- is some sort of semi-automation using LRP
- SpRAy: compute relevance maps (LRP), eigenvalue based spectral clustering -> DR using t-SNE

### A Unified Approach to Interpreting Model Predictions (SHAP values) (Lundberg2017)
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
- model-specific approximation: for linear model just from weights

### On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation (Bach2015)
- **LRP**
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

### Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (Simonyan2014)
- **Gradient** method (seminal, but not important, no need to read)
- showing heatmaps (and cutting according to segmentation)
- showing strongest activating image (deep dreamed or something) for different classes

### Learning how to explain neural networks: PatternNet and PatternAttribution (Kindermans2017)
- other methods fail on linear model -> should "work reliably in the limit of simplicity, the linea models"
- test methods on linear generative data and linear model
- uses same example as _"Theoretical Behavior of XAI Methods in the Presence of Suppressor Variables"_ with supressor/distractor variable
- "PatternAttribution is a DTD extension that learns from data how to set the root point"

### 
____________________________________________________________________

##  2. Causality and XAI

### Causal feature learning: an overview (Chalupka2016)
- "Just about any scientific discipline is concerned with developing ‘macrovariables’ that summarize an underlying finer-scale structure of ‘microvariables’"
- a parallel approach to PCMCI -> not explicitly modelling for time series but just seeing a "cluster" of micro-variables/states as one macro variable
- ambiguous manipulation: "low-cholesterol" -> there is an appropriate level of aggregation

### CAUSALITY FOR MACHINE LEARNING (Schoelkopf2019)
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

### Causal Explanations and XAI (Beckers2022)
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

### Causal Adversarial Network for Learning Conditional and Interventional Distributions (Morraffah2020)
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

### Using Causal Inference to Globally Understand Black Box Predictors beyond Saliency Maps (Reimers2019)
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

### Causes and Explanations: A Structural-Model Approach. Part I: Causes - Pearl (Halpern2002)
- Actual Causes?
- Hume: "We may define a cause to be an object followed by another, ..., where, if the first object had not been, the second never had existed"
- if fire A had not started, fire B would be cause and burned down house should not be "actual cause"
- necessary causality is related to but different from actual causality
- related to specific events: was X = x the actual cause of Y = y in that scenario?
- lots of interesting examples showing how the choice of the SCM is very important

### Causes and Explanations: A Structural-Model Approach. Part II: Explanations - Pearl (Halpern2005)
- The basic idea is that an explanation is a fact that is not known for certain but, if found to be true, would constitute an actual cause of the explanandum (the fact to be explained), regardless of the agent’s initial uncertainty
- the definition of an explanation should be relative to the agent’s epistemic state
- if A is explanation of B happening, after "discovering" A it is not an explanation anymore but part of epistemic state
- we have disallowed disjunctive explanations: there could be many reasons at once
- better explanation: has higher probability of occuring
- partial explanations: victoria went to the canary islands and it was sunny there
- dominant in AI: maximum a posteriori (MAP) approach: best explanation for an observation is the state of the world that is most probable given the evidence

### Counterfactuals uncover the modular structure of deep generative models (Besserve2018)
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

### Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models (Heskes2020)
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

### Causal Interpretability for Machine Learning - Problems, Methods and Evaluation (Morraffah2020a)
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

### Discovering Causal Signals in Images (Lopez-Paz2017)
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

### Instance-wise Causal Feature Selection for Model Interpretation (Panda2021)
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

### Explaining Classifiers with Causal Concept Effect (CaCE) (Goyal2019)
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

### Unsupervised Causal Binary Concepts Discovery with VAE for Black-box Model Explanation (Tran2022)
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

### Explaining Visual Models by Causal Attribution (Parafita2019)
- **causal effect of intervening on generating factors**
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

### Neural Network Attributions: A Causal Perspective (Chattopadhyay2019)
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

### Explaining Deep Learning Models using Causal Inference (Narendra2018)
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

### On the Relationship Between Explanation and Prediction: A Causal View (Karimi2023)
- **also causal method**
- look at treatment effect of prediction on explanation when blocking paths to hyperparameters
- effect is small (probably because hyperparameters and input data make all the difference)
___________________________________________________________________

## 3. Evaluation of Explanation Methods

### Salient Image Net: How to Discover Spurious Features in Deep Learning? (Singla2022)
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

### Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV) (Kim2018)
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

### Ground Truth Evaluation of Neural Network Explanations with CLEVR-XAI (Arras2022)
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

## A Holistic Approach to Unifying Automatic Concept Extraction and Concept Importance Estimation (Fel2023a)
- general framework for concept importance estimation:
1. concept extraction
2. importance estimation
- recast "concept extraction" as "dictionary learning"
- recast "concept importance estimation" as "attribution method"
- dictionary learning: A ~ UV.T (A= original activations, U transformation into "dictionary space", V = dictionary)
- K-Means, PCA, NMF (CRAFT and ICE), Sparse Autoencoder
- ACE -> CAVs = centroids of k-means clusters, but only one concept per sample possible 
- PCA -> lower constraints, but not able to capture "stable concepts" e.g. dog-head if it is important for all dog images, also orthogonality constraint might not be ideal 
- NMF -> moderately sparse representation
- **last layer is best for concept extraction!?**
- Evaluation of Concept Importance Methods:
- C-Deletion, C-Insertion, C-myFidelity
- C-Deletion: gradually remove most important concepts, area under curve
- C-Insertion: gradually add concepts in increasing order of importance
- C-myFidelity: correlation between output when concepts randomly removed 
- concept has *global importance* which can be decomposed into *reliability* (how diagnostic for class) and *prevalence* (how frequently encountered)
- do big (and seemingly well conducted) user study

### Benchmarking Attribution Methods with Relative Feature Importance (Been Kim, Mengjiao Yang) (Yang2019)
- **relative feature importance** similar to wm/(wm+shape) 
- **MCS** model contrast score could be used for wm/shape attribution
- a priori knowledge of relative feature importance
- 1. carefully crafted dataset with known feature importance
- 2. 3 metrics to quantitatively evaluate attribution methods
- certain methods tend to produce *false-positive* (not actually important marked important)
- *relative*: we do not know *absolute* importance of a feature, but can identify *relative* feature importance, compared to another model -> by changing frequency of features in dataset
- metrics compare attributions between pairs of models *model dependence* and pairs of input *input (in)dependence*
- train one model on object and one on scene (background) of images
- take average attribution that saliency method applies to image region (object or background)
- *CF* common feature (or bias) -> with different degree
- one model trained to recognize dogs, one to recognize background
Metrics:
1. Model Contrast Score (MCS): difference between concept attributions between the two models
2. Input dependence rate (IDR): percentage of correctly classified inputs where CF is attributed less than original regions covered by CF ("unimportant feature")
3. Input Independence rate (IIR): How similar is attribution result when dog is present / not present


*2.1. Criticisim of XAI Methods:*

### Theoretical Behavior of XAI Methods in the Presence of Suppressor Variables (Wilming2023)
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

### When Explanations Lie: Why Many Modified BP Attributions Fail (Sixt2020)
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

### A Rigorous Study Of The Deep Taylor Decomposition (Sixt2022)
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

### The (Un)reliability of Saliency Methods (Kindermans2019)
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

### Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations (Locatello2018)
- (for proof that generative stuff is not a good idea, and VAEs)
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


### On the Relationship Between Explanation and Prediction: A Causal View (Karimi2023)
- **also causal method**
- look at treatment effect of prediction on explanation when blocking paths to hyperparameters
- effect is small (probably because hyperparameters and input data make all the difference)

___________________________________________________________________
##  3. Concept Methods and CRP

### Network Dissection: Quantifying Interpretability of Deep Visual Representations (Bau2017)
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

### Understanding the Role of Individual Units in a Deep Neural Network (Bau2020)
- **Network Dissection**
- we wish to understand if it is a spurious correlation, or if the unit has a causal role that reveals how the network models its higher-level notions about trees
- not _where_ network looks (saliency maps) but _what_ it is looking for and _why_
- tasks image classification and image generation
- test the _causal_ structure of network behavior by activating and deactivating the units during processing
- remove e.g. 20 most important units

### Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces (Chormai2022)
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

### Sparse Subspace Clustering for Concept Discovery (SSCCD) (Vielhaben2022)
- motivation: reveal coherent, discriminative structures exploited by model, rather than accordance with human-identified concepts
- concepts not one- but low-dimensional subspaces
- approach does not require reinforcing "interpretability" e.g. enforcing dissimilarity of concepts
- concept interpretability method: a. concept discovery, b. mapping feature - input space c. relevance quantification
- for b. translate feature level concept maps to the input level by simple bilinear interpolation

### Revealing Hidden Context Bias in Segmentation and Object Detection Through Concept-Specific Explanations (Dreyer2023)
- **application of CRP for segmentation and object detection**
- for segmentation: "heatmap" just resembles segmentation itself (just tresholded)
- glocal/ CRP: _where_ the model pays attention to and _what_ it sees
- "Especially in the multi-object context, it is crucial to attain object-class-specific explanations, which is not possible by the analysis of latent activations: Concepts with the highest activation values can refer to any class that is present in the image, such as the horse’s rider, or none at all, since activations do not indicate whether a feature is actually used for inference."
- basically same content as main paper, showing how to use concept relevance maximization to visualize features and find biases/context dependency -> "context bias"

### XAI-TRIS: Non-linear benchmarks to quantify ML explanation performance (Benedict Clark, Rick Wilming, Stefan Haufe) (Clark2023)
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

### Do Users Benefit From Interpretable Vision? A User Study, Baseline, And Dataset (Sixt2022a)
- compare "baseline" (just predictions ordered by confidence) with some other XAI methods
- other methods:
  - "Invertible Neural Networks": moving along weight vector changes prediction
  - "Concepts (Zhang)": automatically-discovered concepts (NMF approach), non-negative matrix factorization -> non-negative TCAVs
- claim: first to extensively test counterfactual-based and concept-based explanations on bias discovery using a challenging dataset
- Spatially overlapping attributes, like color and shape, directly challenge saliency map explanations.
- might be good measure for ground-truth feature importance: "PREDICTION FLIP" measure difference in prediction with feature on/off
- calculated a linear fit for each parameter change to the logit change. We reported the coefficient of determination R2 , which indicates how much of the variance in the prediction can be explained linearly by the analyzed property

### From attribution maps to human-understandable explanations through Concept Relevance Propagation (Achtibat2023)
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

### When are Post-hoc Conceptual Explanations Identifiable? (Leeman2023)
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

### Multi-dimensional concept discovery (MCD): A unifying framework with completeness guarantees (Vielhaben2023)
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