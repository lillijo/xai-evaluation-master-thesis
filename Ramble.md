# RAMBLE DOWN STUFF

## Ramble that almost sounds scientific
We believe that this approach is the first that tries to not see biasedness of a data distribution and the learned model to be a binary variable. Instead we take the relationship between the spurious and the core feature to be a variable which can be causally intervened on continously. 

What is mostly not mentioned in papers evaluating how well attribution methods find clever hans behavior, is how strong the effect of random initialization is to the model. In our experiments we found that while there was a clear link between the spurious-to-core feature ratio and the models sensitivity to the watermark, still even the most biased models often managed to ignore the spurious feature all together. In contrast to other work (cite) which sets the true data distribution equal to the models prediction, we specifically acknowledge the non-linear relationship between the data distribution and what the model really learns.
Similar to Karimi2023 we have found the effect of random initialization on the model and hence the explanation to be way stronger than expected in comparison to the biasedness. 

It is weird that many papers introducing new explanation methods or evaluating them implicitly use causal methods without specifically pointing them out. I believe that AI scientists have become so disincentivized using the word "causal" for anything because clearly models learn only correlations and not causal models. However there is nothing wrong with using causal language when examining explanations and the inner workings of neural networks. Indeed, many authors (cite) have pointed out how straight-forward it is to interpret neural networks as structural causal models, not forgetting that all the seminal work on bayesian models from Pearl and many others was taken as inspiration in their very construction. 

In recent years the number of causality-inspired explanations and their evaluation has grown substantially. As causal analysis necessitates the availability of a ground-truth causal model most of the work has to fall back to quite trivial benchmark datasets with known generating factors. Allthough it is definitely possible to make general remarks about the limits of neural networks as well as their explanations with such a setup, it ignores the real application case for complex models: complex image and language data without known ground truths. 

The idea of this experiment therefore is to gather evidence on the applicability of evaluation methods to recognize causal structure within explanations without a ground-truth. Of course, this again requires a benchmark dataset with known causal model. Being aware of not only the true data generation but also the true reaction of the model to intervening on each generating factor is vital to establish whether any causal claims can be made by explanation models.
In a lot of work the chain "data-model-explanation" was simplified to "data/model-explanation" but we believe that this simplifiction poses the risk of conflating either the abilities or downsides of an explanation method, depending on the combination. 


## Ramble for myself
I ask myself why I had to choose such a simple benchmark dataset, because again, it is not realistic and does not model the true application case of the models and explanation methods in question well. But it is important to start the process of developing an idea with the simplest possible model. What is the simplest possible, is obviously up to debate and something I personally struggle with a lot. 

Why does CRP use zero-masking for their conditional attributions??? This completely ignores the interactions concepts or neurons might have with each other. Bluecher2022 tries to address this issue with a new approach deeply rooted in probability theory and related to shapely values. I believe that something similar could be done with CRP. Right now, the zero-masking of hundreds of connections in a neural network, if we were to see it as an SCM, is detrimental to making causal claims on the resulting saliency maps. 

Incidentally I discovered one model that seemed to highlight what this approach can miss:
The model had a very good accuracy of about 99.5% and performed only slightly worse on the ellipse class (~99.3%), while ignoring the watermark completely (bias was only at 0.3 so this is to be expected). I was made aware of an irregularity through looking at the "prediction flip" or "mean logit change", where the shape seemed to have an unusally low effect on this particular model. This came as a surprise because the models prediction was still very good. When I therefore looked at the attribution maps for all different neurons, I could find that while they had differing "relevances" the ellipse itself had very negative relevance. Indeed there was not a single neuron with positive relevance for the ellipse, only some saliency maps that did not seem to assign any relevance to any picture, giving the impression that the shape actually did not play a role in the prediction.
The only explanation I can find for this, is that the model only positively predicts rectangles (because they had more expected saliency maps).

In initial experiments it became clear how easy it is to introduce accidental biases to the data. To increase the complexity of the problem, we had put a small gaussian noise onto each pixel after potentially adding the watermark. Interestingly, the accuracy of the models was very high (> 99%), but when looking at the "causal" interaction between generating factors and the prediction, we were surprised.

I still have not found a good way to summarize the explanations reaction to bias in one measure. The whole idea of CRP is lost, if we just average over the single neurons/concepts. Instead, we need to find a way to substain the level to which the explanation was able to disentangle the core and spurious features influence on the prediction.
One can of course compute the mean logit change when intervening on each generating factor (watermark, shape, scale, rotation...) individually for each neuron. 

Most work so far that has tested the effect of clever-hans features, watermarks or correlated background has seen this spurious feature as a binary feature. Either the true core feature is spuriously correlated with some other feature or not. However it might be interesting to see biasedness as a continous variable. This not only makes it possible to study the effect of spurious features more accurately with causal methods, it also for the first time enables comparing the importance of the spurious feature between models that use it to varying degree. For example, Yang2019 tried has a related approach to this thesis but only looks at models either learning the one feature or the other and not fine-grained. We believe this continuous approach to better reflect real data, as very often the question is not *is our data biased or not* (spoiler-alert: it always is) but *is our data unbiased enough to differentiate important from unimportant*. 

Why do we want the discovery of the spurious feature to be more unsupervised? Mostly because we hope that if it is easy to recover without having the ground truth data available, this speaks for the disentanglement to also be obvious for humans when looking at the concepts and their heatmaps or maximally relevant sample sets. Though it has been shown that PCA and similar dimensionality reduction algorithms are not good at finding disentagled latent factors


Similar to Karimi2023 we do not just assume that model training follows some iid distribution. Instead, we intervene or fix each of the hyperparameters ever used. We do this by constructing a causal graph, including all knowledge we have of the trained model.
Then one can do causal discovery using this knowledge. We have an assumed true graph, where watermark and shape have causal effects depending on ratio r, and the trained model weights are also depending on the random initialization of the model. 
If CRP and LRP work as expected, they would reflect the effect of the random initialization accurately when the whole generating graph is marginalized out. 
But this is not possible with the true image, as without it no CRP or LRP can happen. 

We want to measure the causal effect of the (learned) model weights on the explanation. The generating information as well as the seed act as instrumental variables according to Pearls Front-door criterion. 

# good words for "bias"
- signal-to-noise ratio - SNR - rho - R
- spurious-to-core (feature) ratio
- biasedness
- strength of spurious correlation
- confounder effect size/ratio
- confounded-noise-to-signal ratio



Pipeline:

1. construct a structural causal model linking the existing generating factors of the dataset
    - see image, but also try with different models later on
2. sample many datasets by intervening on the bias or "spurious-to-core feature ratio" from 0 to 1, 0 meaning the confounder has 0 effect, 1 meaning shape and watermark are completely correlated
    - 
3. train the same simple model architecture with fixed hyperparameters (only random initialization changes???) on the intervened datasets 
4. establish ground-truth importance by estimating the causal effect of latent generating factors on model prediction
5. extract CRP attributions for a small set of images
6. Estimate the causal effect of generating factors on CRP conditional attribution (in different layers)
7. Attempt to disentangle attributions and identify human understandable concepts (shape, watermark) 
8. Evaluate whether the CRP attributions rely stronger on the causal effect of the latent factors or stronger on what the model has actually learned. Also evaluate whether the CRP "concepts" can be disentangled into human understandable concepts relating to the true generating factors



Theory:
- normal attribution methods can only assign vague importance to all parts of image
- CRP / neuron concepts have potential to say "shape is relevant, watermark not so much" or opposite, or "watermark is relevant for class 1 but not 0"
- question: how good is plain old LRP at scoring relevance importance, and how much better is CRP at it
- answer: take watermark bbox importance of LRP / general heatmap, and of each neuron individually
- what is explicit information gain? (or is there even one)
-> is error between CRP importance and true importance larger or smaller than LRP importance? 

MEASURE:
- CRP: watermark bounding box importance for N most relevant individual neurons -> weighted by relevance of neuron
- LRP: watermark bounding box importance in summary image

What do i want to measure?

- disentangledness - can CRP differentiate between watermark and shape?
- importance of watermark feature, in relation to shape feature
- negative vs positive relevance - does class speficific approach fail?
- 

what is "good" outcome?
- do we want to explain the data or the model? depends on what we are comparing to. Mostly we have same data but different model (or?)
- so we want to compare (and explain) the model

Problem: we cannot compare heatmap similarity/correlation over multiple models 
Solution: have an abstracted measure of importance for explanation - like  watermark bbox importance, NMF vectors or similar


MEASURES FOR CRP:

- general mean relevance change for watermark
- general predictive value of generating factors to relevances (Ordinary Least Squares R2 Norm)
- sum of n most relevant neurons inside watermark bounding box (only for images with watermark?)
- mean activation change vs mean relevance change for watermark?
- relevance mass accuracy (percentage of relevance within bbox) (clever-xai)
- relevance rank accuracy (#pixels in bbox which are within n most important pixels)


- questions