
# Multimodal Learning and Large Language Models
## A GitHub-Ready Textbook Draft

> This document is written as a long-form textbook-style Markdown draft for a GitHub repository.
> Organization: **(1) Multimodal Learning** → **(2) Large Language Models (LLMs)**.
> The emphasis is on **background knowledge, classical foundations, representative equations, architectural evolution, practical design choices, limitations, and current trends**.
> Mathematical symbols are explained immediately after each equation so that the flow remains readable for learners.

---

# Part I. Multimodal Learning

## 1. Why multimodal learning matters

Humans rarely reason from a single source of information. We combine language, vision, sound, touch, time, and context. Machine learning faces a similar reality. Many real-world problems are not naturally unimodal:

- medical AI: image + report + laboratory values + longitudinal records
- autonomous systems: camera + lidar + radar + map + text commands
- human-centered AI: speech + facial expression + gesture + text
- industrial AI: sensor streams + images + maintenance logs + operating conditions
- recommendation/search: image + title + description + user behavior

A **modality** means one type of signal or representation channel, such as text, image, audio, video, tabular data, biosignals, or graphs.  
**Multimodal learning** studies how to represent, align, fuse, generate, or reason across multiple modalities.

A useful intuition is:

- **representation problem**: how do we encode each modality?
- **alignment problem**: how do we connect semantically related signals across modalities?
- **fusion problem**: how do we combine them for prediction or generation?
- **robustness problem**: what if one modality is noisy, missing, delayed, or contradictory?
- **generalization problem**: does fusion improve performance outside the training distribution?

---

## 2. Problem settings in multimodal learning

Multimodal learning is not one single task. It includes several related settings.

### 2.1 Representation learning
Learn useful latent features from multiple modalities.

### 2.2 Cross-modal alignment
Map different modalities into a shared or coordinated space so that related samples become close.

Examples:
- image ↔ caption
- speech ↔ text transcript
- MRI ↔ radiology report
- sensor signal ↔ fault description

### 2.3 Fusion for prediction
Combine modalities to predict a target:
- classification
- regression
- ranking
- survival analysis
- anomaly detection

### 2.4 Cross-modal generation
Generate one modality from another:
- text from image
- image from text
- audio from text
- missing signal reconstruction from observed signals

### 2.5 Retrieval
Retrieve across modalities:
- text-to-image retrieval
- image-to-text retrieval
- report-to-scan retrieval

### 2.6 Missing modality learning
Train or test when some modalities are absent, which is very common in healthcare and industrial settings.

---

## 3. Historical development

### 3.1 Before deep learning
Earlier multimodal systems often depended on:
- hand-crafted features
- probabilistic graphical models
- canonical correlation analysis (CCA)
- late score fusion
- modality-specific pipelines stitched together manually

These systems could work well in constrained settings, but they struggled when modalities were high-dimensional and strongly nonlinear.

### 3.2 Early deep multimodal learning
A key early milestone was **Multimodal Deep Learning** by Ngiam et al. (2011), which showed that deep networks could learn joint features from audio and video and even support cross-modal feature learning. This helped establish deep multimodal representation learning as a practical direction.

### 3.3 Representation alignment era
Methods such as **Deep Canonical Correlation Analysis (DCCA)** extended classical correlation-based learning into nonlinear deep architectures.

### 3.4 Transformer and contrastive era
After the Transformer, multimodal learning accelerated quickly:
- attention-based fusion
- image–text contrastive learning
- large-scale web supervision
- instruction tuning for vision–language systems
- large multimodal models with in-context behavior

---

## 4. Classical foundation: Canonical Correlation Analysis (CCA)

CCA is one of the most important mathematical starting points for multimodal learning.

Suppose we observe two views of the same object:

- \(x \in \mathbb{R}^{d_x}\): first modality
- \(y \in \mathbb{R}^{d_y}\): second modality

CCA seeks linear projections \(w_x\) and \(w_y\) such that the projected variables are maximally correlated.

\[
\max_{w_x, w_y} \ \mathrm{corr}(w_x^\top x,\; w_y^\top y)
\]

Expanded form:

\[
\max_{w_x, w_y}
\frac{w_x^\top \Sigma_{xy} w_y}
{\sqrt{w_x^\top \Sigma_{xx} w_x}\sqrt{w_y^\top \Sigma_{yy} w_y}}
\]

### Meaning of each symbol
- \(w_x\): projection vector for modality \(x\)
- \(w_y\): projection vector for modality \(y\)
- \(\Sigma_{xx}\): covariance matrix of modality \(x\)
- \(\Sigma_{yy}\): covariance matrix of modality \(y\)
- \(\Sigma_{xy}\): cross-covariance matrix between \(x\) and \(y\)
- numerator \(w_x^\top \Sigma_{xy} w_y\): covariance between the projected variables
- denominator: product of projected standard deviations
- overall objective: maximize correlation, not merely raw covariance

### Why CCA matters
CCA answers a foundational multimodal question:

> “Can we find two transformations of two modalities such that they emphasize only the information they share?”

That idea reappears in many modern multimodal methods.

### Limitation
CCA is linear. If image-text or audio-video relations are nonlinear, linear CCA is often insufficient.

---

## 5. Deep Canonical Correlation Analysis (DCCA)

DCCA replaces linear projections with neural networks.

Let
- \(f_\theta(x)\): nonlinear encoder for modality \(x\)
- \(g_\phi(y)\): nonlinear encoder for modality \(y\)

Then DCCA solves

\[
\max_{\theta,\phi} \ \mathrm{corr}(f_\theta(x),\; g_\phi(y))
\]

### Meaning
- \(\theta\): parameters of the first neural network
- \(\phi\): parameters of the second neural network
- \(f_\theta(x)\): learned representation of the first modality
- \(g_\phi(y)\): learned representation of the second modality
- the correlation is computed between the learned embeddings, not the raw inputs

### Why it was important
DCCA kept the spirit of CCA while allowing nonlinear feature extraction. It became a conceptual bridge from classical multiview statistics to modern deep multimodal learning.

---

## 6. Three main fusion strategies: early, late, and hybrid

A simple but powerful taxonomy for multimodal prediction is the following.

### 6.1 Early fusion
Concatenate or combine features before the predictor.

\[
z = [h_1; h_2; \dots; h_M]
\]

- \(h_m\): representation of the \(m\)-th modality
- \([\,;\,]\): concatenation operator
- \(z\): joint representation fed into a downstream model

Advantages:
- simple
- allows feature interactions downstream

Limitations:
- sensitive to dimensional imbalance
- may fail if modalities have different noise levels, sampling rates, or missingness

### 6.2 Late fusion
Make separate predictions and combine them.

If modality-specific predictions are \(p_1, p_2, \dots, p_M\), then one simple late-fusion rule is

\[
\hat{p} = \sum_{m=1}^{M} \alpha_m p_m
\]

with

\[
\sum_{m=1}^{M} \alpha_m = 1,\quad \alpha_m \ge 0
\]

Meaning:
- \(p_m\): prediction from modality \(m\)
- \(\alpha_m\): weight assigned to modality \(m\)
- \(\hat{p}\): fused prediction

Advantages:
- robust
- easy to interpret
- useful when modalities differ in reliability

Limitations:
- may miss rich cross-modal interactions

### 6.3 Hybrid/intermediate fusion
Fuse at intermediate representation layers, often with cross-attention, gating, tensor fusion, or hierarchical fusion.

This is often the most expressive regime in modern architectures.

---

## 7. Representation learning objectives

Modern multimodal systems usually train encoders with one or more of the following objectives.

### 7.1 Supervised task loss
For classification:

\[
\mathcal{L}_{\text{cls}} = - \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic}\log \hat{y}_{ic}
\]

Meaning:
- \(N\): number of training samples
- \(C\): number of classes
- \(y_{ic}\): true label indicator for sample \(i\) and class \(c\)
- \(\hat{y}_{ic}\): predicted probability for class \(c\)
- the loss penalizes confident wrong predictions

For regression, common losses include

\[
\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

Meaning:
- \(y_i\): true target
- \(\hat{y}_i\): predicted target
- squared error emphasizes larger mistakes more strongly

### 7.2 Reconstruction loss
If the model tries to reconstruct one or more modalities:

\[
\mathcal{L}_{\text{rec}} = \|x - \hat{x}\|_2^2
\]

Meaning:
- \(x\): original signal
- \(\hat{x}\): reconstructed signal
- \(\|\cdot\|_2^2\): squared Euclidean norm

This is common in autoencoder-style multimodal learning and missing-modality recovery.

### 7.3 Contrastive alignment loss
A cornerstone of modern image-text learning is contrastive training.

For a minibatch of matched image-text pairs \((v_i, t_i)\), define similarity

\[
s_{ij} = \frac{\langle z_i^{(v)}, z_j^{(t)} \rangle}{\tau}
\]

Meaning:
- \(z_i^{(v)}\): image embedding for sample \(i\)
- \(z_j^{(t)}\): text embedding for sample \(j\)
- \(\langle \cdot,\cdot \rangle\): dot product
- \(\tau\): temperature, controlling how sharp the softmax becomes

Then the image-to-text contrastive loss is

\[
\mathcal{L}_{v\rightarrow t}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ij})}
\]

And the text-to-image direction is

\[
\mathcal{L}_{t\rightarrow v}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{N}\exp(s_{ji})}
\]

The full bidirectional loss is

\[
\mathcal{L}_{\text{contrastive}}
=
\frac{1}{2}
\left(
\mathcal{L}_{v\rightarrow t}
+
\mathcal{L}_{t\rightarrow v}
\right)
\]

Interpretation:
- the numerator uses the matched pair \((i,i)\)
- the denominator compares that match against all candidates in the batch
- the objective pushes matched pairs together and mismatched pairs apart

This is the core idea behind CLIP-style learning.

---

## 8. Fusion mechanisms in more detail

### 8.1 Concatenation
The simplest form:
\[
z = [h_1;h_2;\dots;h_M]
\]

Works surprisingly well as a baseline and should never be skipped in empirical studies.

### 8.2 Gated fusion
A learned gate decides how much each modality should contribute.

For two modalities:

\[
g = \sigma(W_g[h_1;h_2] + b_g)
\]

\[
z = g \odot h_1 + (1-g)\odot h_2
\]

Meaning:
- \(W_g, b_g\): gate parameters
- \(\sigma(\cdot)\): sigmoid function, outputting values in \((0,1)\)
- \(g\): gate vector
- \(\odot\): element-wise multiplication
- each component of \(g\) determines whether the representation relies more on modality 1 or modality 2

This is useful when modality reliability varies by sample.

### 8.3 Bilinear or tensor fusion
For richer interactions, the model can explicitly model multiplicative cross-modal terms.

A bilinear score is

\[
f(h_1,h_2)=h_1^\top W h_2
\]

Meaning:
- \(h_1\): representation of modality 1
- \(h_2\): representation of modality 2
- \(W\): parameter tensor/matrix controlling pairwise interactions
- unlike simple concatenation, bilinear fusion captures multiplicative relationships

Tensor Fusion Networks extended this idea for multimodal sentiment analysis by modeling unimodal, bimodal, and trimodal interactions.

### 8.4 Cross-attention
Cross-attention is now one of the most central multimodal fusion tools.

Given query matrix \(Q\), key matrix \(K\), and value matrix \(V\),

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Meaning:
- \(Q\): queries, asking “what should I attend to?”
- \(K\): keys, describing candidate items
- \(V\): values, the actual information to aggregate
- \(d_k\): key dimension, used for scaling
- \(QK^\top\): similarity between queries and keys
- softmax: converts similarities into attention weights
- multiplying by \(V\): weighted aggregation of information

In multimodal settings:
- text queries can attend to image keys/values
- image tokens can attend to language tokens
- one biosignal stream can attend to another sensor stream

Cross-attention is especially powerful because it supports:
- variable-length inputs
- alignment without strict temporal synchronization
- interpretable token-to-token interactions

---

## 9. Multimodal transformers

The Transformer changed multimodal learning for two main reasons:

1. it provides a general-purpose sequence-processing architecture
2. attention naturally supports interactions across modalities

### 9.1 Self-attention
Within one modality:

\[
\mathrm{SelfAttn}(X)=\mathrm{softmax}\left(\frac{XW_Q(XW_K)^\top}{\sqrt{d_k}}\right)XW_V
\]

Meaning:
- \(X\): token matrix for one modality
- \(W_Q, W_K, W_V\): learned projections for queries, keys, values

### 9.2 Cross-modal attention
When modality A attends to modality B:

\[
\mathrm{CrossAttn}(X_A, X_B)
=
\mathrm{softmax}
\left(
\frac{(X_AW_Q)(X_BW_K)^\top}{\sqrt{d_k}}
\right)
(X_BW_V)
\]

Meaning:
- \(X_A\): tokens from the querying modality
- \(X_B\): tokens from the source modality being attended to
- the output is a modality-A representation enriched by information from modality B

This idea underlies many modern multimodal architectures.

---

## 10. Contrastive multimodal learning and the rise of CLIP-style models

A major turning point came from large-scale image-text contrastive pretraining.

### 10.1 Shared embedding space
Image encoder \(f_v\) and text encoder \(f_t\) produce embeddings:

\[
z^{(v)} = \frac{f_v(v)}{\|f_v(v)\|_2},
\qquad
z^{(t)} = \frac{f_t(t)}{\|f_t(t)\|_2}
\]

Meaning:
- each encoder maps its modality into a vector space
- \(L_2\)-normalization ensures unit norm
- cosine similarity becomes equivalent to dot product

Then the model is trained so matched image-text pairs are close.

### 10.2 Why CLIP mattered
CLIP showed that large-scale natural language supervision can yield strong zero-shot transfer. Instead of training a classifier only for fixed labels, the model can compare an image against text prompts like:
- “a photo of a cat”
- “a chest X-ray showing pneumonia”
- “a defective wafer pattern”

This changed the role of language from output-only to supervision, interface, and retrieval key.

### 10.3 Practical lesson
Large-scale multimodal training often benefits more from:
- large and diverse data
- well-designed contrastive objectives
- strong encoders
than from excessively complicated fusion blocks.

---

## 11. Generative multimodal modeling

Contrastive models align modalities, but generative models learn to produce one modality from another.

### 11.1 Conditional generation
A general conditional modeling objective is

\[
p(y \mid x)
\]

where
- \(x\): conditioning modality or modalities
- \(y\): modality to generate

Examples:
- caption generation: \(p(\text{text} \mid \text{image})\)
- image synthesis: \(p(\text{image} \mid \text{text})\)
- report generation: \(p(\text{report} \mid \text{medical image})\)

### 11.2 Autoregressive sequence modeling
If the output is tokenized as \(y_1,\dots,y_T\), then

\[
p(y \mid x) = \prod_{t=1}^{T} p(y_t \mid y_{<t}, x)
\]

Meaning:
- \(y_t\): token at step \(t\)
- \(y_{<t}\): all previous tokens
- the model generates one token at a time conditioned on the input and prior outputs

This is the backbone of many multimodal captioning and instruction-following systems.

### 11.3 Diffusion-style multimodal generation
For image generation, diffusion models learn to reverse a noise-adding process. While the full derivation is beyond a short summary, the key idea is:
- gradually add noise during forward diffusion
- train a network to predict or remove noise during reverse diffusion
- condition the denoising process on text or other modalities

This made text-to-image generation practical at high visual fidelity.

---

## 12. Missing modality learning

This topic is especially important in medicine, industry, and field deployment.

Typical causes:
- sensor failure
- cost constraints
- privacy restrictions
- irregular acquisition
- absent reports or annotations
- institution-specific protocols

### 12.1 Training-time missingness vs test-time missingness
Two cases should be distinguished:

1. **training-time missingness**: some modalities are absent during training  
2. **test-time missingness**: model is trained with all modalities, but deployment inputs may omit some

The second case often causes severe performance drop if it was ignored in training.

### 12.2 Common solutions
- modality dropout during training
- generative imputation
- cross-modal distillation
- shared latent spaces
- mixture-of-experts style routing
- uncertainty-aware fusion
- separate models for subsets of modalities

### 12.3 Modality dropout
A simple idea is to randomly drop modalities during training so the network learns to survive partial observations.

If \(m_k \in \{0,1\}\) is a binary indicator for modality \(k\), then a fused representation may be written as

\[
z = \mathrm{Fuse}(m_1 h_1,\dots,m_M h_M)
\]

Meaning:
- \(m_k=1\): modality present
- \(m_k=0\): modality masked
- the model learns robustness to missing modalities

This is conceptually simple and often very effective.

---

## 13. Synchronization and alignment issues

Not all multimodal data are neatly aligned.

Examples:
- video at 30 FPS, audio at 16 kHz, text at sentence level
- EMR timestamps irregular, labs sparse, imaging occasional
- multiple industrial sensors sampled at different frequencies

This creates two major challenges:
- **temporal misalignment**
- **asynchronous semantics**

Solutions include:
- interpolation/resampling
- hierarchical encoders
- dynamic time warping variants
- cross-attention over non-aligned sequences
- latent-time models
- continuous-time representations

MulT is a representative example showing that explicit strict alignment is not always necessary if crossmodal attention can learn interactions across time.

---

## 14. Evaluation in multimodal learning

Evaluation must match the task.

### 14.1 Classification/regression
Use standard metrics:
- accuracy
- AUROC
- F1-score
- MAE
- RMSE
- calibration metrics

### 14.2 Retrieval
Typical metrics:
- Recall@K
- mean reciprocal rank
- median rank

### 14.3 Generation
Typical metrics:
- BLEU
- ROUGE
- CIDEr
- SPICE
- human evaluation
- task-grounded evaluation

### 14.4 Robustness
A multimodal model should not be judged only by average benchmark score.

Important additional tests:
- single-modality ablation
- missing-modality evaluation
- corrupted modality evaluation
- out-of-distribution testing
- cross-site or cross-device validation
- fairness and subgroup robustness
- calibration under modality conflict

---

## 15. Representative model families in multimodal learning

### 15.1 Correlation-based models
- CCA
- DCCA
- generalized CCA variants

Strength:
- elegant mathematical interpretation

Limitation:
- less expressive than large-scale contrastive/generative systems

### 15.2 Fusion networks
- concatenation MLPs
- gated fusion
- bilinear pooling
- tensor fusion

Strength:
- strong on supervised multimodal prediction

### 15.3 Transformer-based multimodal sequence models
- self-attention and cross-attention architectures
- aligned or unaligned sequence modeling
- flexible and scalable

### 15.4 Contrastive dual encoders
- CLIP
- ALIGN
- retrieval-oriented models

Strength:
- scalable web supervision
- strong retrieval and zero-shot transfer

### 15.5 Large vision-language / multimodal instruction models
- Flamingo
- BLIP / BLIP-2
- PaLI
- LLaVA family
- Gemini-style long-context multimodal systems

Strength:
- open-ended generation and reasoning over image-text, and in some cases video/audio too

---

## 16. Representative papers every textbook should mention

Below is a compact reading path.

### Foundational and classical
1. Ngiam et al., **Multimodal Deep Learning** (2011)  
2. Andrew et al., **Deep Canonical Correlation Analysis** (2013)

### Fusion and multimodal language
3. Zadeh et al., **Tensor Fusion Network for Multimodal Sentiment Analysis** (2017)  
4. Tsai et al., **Multimodal Transformer for Unaligned Multimodal Language Sequences** (2019)

### Large-scale contrastive image-text learning
5. Radford et al., **CLIP** (2021)  
6. Jia et al., **ALIGN** (2021)

### Large multimodal generation / instruction-following
7. Alayrac et al., **Flamingo** (2022)  
8. Li et al., **BLIP** (2022)  
9. Li et al., **BLIP-2** (2023)  
10. Chen et al., **PaLI** (2022)  
11. Liu et al., **LLaVA / Visual Instruction Tuning** (2023)

### Recent directions
12. Gemini Team, **Gemini 1.5 Technical Report** (2024)  
13. Wu et al., **Deep Multimodal Learning with Missing Modality: A Survey** (2024)

---

## 17. Current trends in multimodal learning

### 17.1 Large multimodal models (LMMs)
The field has shifted from task-specific multimodal models to general-purpose multimodal models that accept mixed inputs and produce flexible outputs.

### 17.2 Longer context windows
Modern systems increasingly process:
- multiple images
- video clips
- long documents
- long audio
- interleaved multimodal conversations

### 17.3 Instruction tuning
A major shift has been from “predict labels” to “follow instructions.”  
This makes multimodal systems more usable as interfaces.

### 17.4 Retrieval-augmented multimodal systems
External memory and retrieval reduce hallucination and improve provenance.

### 17.5 Missing-modality robustness
This is becoming more central in healthcare, autonomous systems, and edge deployment.

### 17.6 Data quality over raw scale
As scale grows, data filtering, deduplication, and balancing matter more.

### 17.7 Evaluation beyond leaderboard averages
There is increasing emphasis on:
- factuality
- calibration
- safety
- bias
- reliability under distribution shift
- domain-specific trustworthiness

---

## 18. Open research challenges in multimodal learning

1. **What is real fusion?**  
   Sometimes extra modalities help only because they act as shortcuts or proxies.

2. **Missing and conflicting modalities**  
   Real data are incomplete and disagreeing, not perfectly synchronized.

3. **Spurious cross-modal correlations**  
   A report style or acquisition device may leak labels.

4. **Interpretability**  
   Attention maps alone do not automatically prove causal use of a modality.

5. **Benchmark mismatch**  
   Many benchmarks are much cleaner than real deployment settings.

6. **Efficient adaptation**  
   Large multimodal models are expensive to fine-tune and evaluate.

7. **Fairness and privacy**  
   More modalities can increase both utility and privacy risk.

---

## 19. Practical design guidelines for researchers

When building a multimodal model, a disciplined experimental order is useful:

### Step 1. Build unimodal baselines first
You must know whether multimodality truly helps.

### Step 2. Add simple late and early fusion baselines
Do not jump immediately to cross-attention.

### Step 3. Study missing-modality robustness explicitly
Test all subsets that matter in deployment.

### Step 4. Audit spurious correlations
Check whether one modality encodes institution, device, or demographic proxies.

### Step 5. Report calibration and subgroup performance
Average AUC alone is rarely enough.

### Step 6. Use ablation studies
Remove modalities, fusion blocks, alignment losses, and pretraining stages one by one.

---

# Part II. Large Language Models (LLMs)

## 20. What is an LLM?

A **large language model** is a neural network, usually Transformer-based, trained on massive text corpora to model language patterns. In practice, an LLM learns to perform next-token prediction or related self-supervised objectives, then is often adapted with instruction tuning, preference optimization, retrieval, or tool use.

At a high level, an LLM estimates probabilities like

\[
p(x_1, x_2, \dots, x_T)
\]

which, by the chain rule, becomes

\[
p(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T} p(x_t \mid x_{<t})
\]

### Meaning
- \(x_t\): the token at position \(t\)
- \(x_{<t}\): all previous tokens before position \(t\)
- the model predicts each next token conditioned on earlier tokens

This apparently simple objective turned out to produce surprisingly general capabilities when scaled.

---

## 21. Historical path to LLMs

### 21.1 Statistical language models
Before neural LMs, NLP often used n-gram models.

A trigram model approximates

\[
p(x_t \mid x_{<t}) \approx p(x_t \mid x_{t-2}, x_{t-1})
\]

This works locally but cannot model long dependencies well.

### 21.2 Recurrent neural networks and LSTMs
RNNs improved sequence modeling by maintaining hidden states across time, but long-range dependency and training efficiency remained difficult.

### 21.3 Attention and the Transformer
The Transformer removed recurrence and relied on attention, enabling efficient parallel training and better long-context modeling.

### 21.4 Pretrained language models
BERT popularized deep bidirectional pretraining for understanding tasks. GPT-style models popularized autoregressive pretraining for generation.

### 21.5 Scaling era
GPT-3 showed that sufficiently large autoregressive language models can perform many tasks from prompts alone, without parameter updates.

### 21.6 Alignment and instruction-following era
InstructGPT showed that post-training with human feedback can make models more helpful and aligned to user intent.

---

## 22. Tokenization

LLMs do not read raw text directly. They operate on tokens.

Given text sequence \(s\), a tokenizer maps it to
\[
s \mapsto (x_1, x_2, \dots, x_T)
\]

Each \(x_t\) is an integer token ID.

Why tokenization matters:
- vocabulary size affects efficiency
- multilingual coverage depends on token design
- domain-specific language may fragment badly
- context length is measured in tokens, not words

---

## 23. Embeddings and positional information

Each token ID is mapped to a vector:

\[
e_t = E[x_t]
\]

Meaning:
- \(E\): embedding matrix
- \(x_t\): token ID
- \(e_t\): dense vector for token \(t\)

Since Transformers do not inherently know order, positional information is added:

\[
h_t^{(0)} = e_t + p_t
\]

Meaning:
- \(h_t^{(0)}\): input representation at layer 0
- \(e_t\): token embedding
- \(p_t\): positional embedding or positional encoding

Without position, the model would know which tokens are present, but not in which order.

---

## 24. Self-attention in LLMs

This is the core operation behind Transformers.

Given token matrix \(X \in \mathbb{R}^{T \times d}\), compute

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

Meaning:
- \(T\): sequence length
- \(d\): hidden dimension
- \(W_Q, W_K, W_V\): learned projection matrices
- \(Q\): queries
- \(K\): keys
- \(V\): values

Then attention is

\[
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

### Step-by-step interpretation
1. \(QK^\top\) computes similarity between all query-token and key-token pairs.
2. Divide by \(\sqrt{d_k}\) to stabilize scale.
3. Apply softmax row-wise to obtain attention weights.
4. Use those weights to form weighted sums of the values \(V\).

In an autoregressive model, causal masking ensures token \(t\) cannot attend to future tokens \(>t\).

---

## 25. Multi-head attention

Instead of learning only one attention pattern, the model uses several heads:

\[
\mathrm{MultiHead}(X)
=
\mathrm{Concat}(\text{head}_1,\dots,\text{head}_H)W_O
\]

where

\[
\text{head}_h = \mathrm{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})
\]

Meaning:
- \(H\): number of attention heads
- each head can specialize in different relational patterns
- \(W_O\): output projection after concatenating all heads

This increases representational flexibility.

---

## 26. Feed-forward layers and residual blocks

A Transformer block usually contains:
1. multi-head attention
2. position-wise feed-forward network
3. residual connections
4. layer normalization

A typical feed-forward layer is

\[
\mathrm{FFN}(x)=W_2 \sigma(W_1 x + b_1)+b_2
\]

Meaning:
- \(x\): hidden state
- \(W_1, W_2\): learned weight matrices
- \(b_1, b_2\): biases
- \(\sigma\): activation function such as GELU

Residual connection:

\[
h^{(\ell+1)} = h^{(\ell)} + F(h^{(\ell)})
\]

Meaning:
- \(h^{(\ell)}\): representation at layer \(\ell\)
- \(F(\cdot)\): sub-layer transformation
- residual addition helps optimization and gradient flow

---

## 27. Two major pretraining paradigms: encoder-style and decoder-style

### 27.1 Encoder-style masked language modeling (BERT-like)
BERT masks some tokens and predicts them from both left and right context.

If \(\mathcal{M}\) is the set of masked positions, the objective is

\[
\mathcal{L}_{\text{MLM}}
=
-\sum_{t \in \mathcal{M}} \log p(x_t \mid x_{\setminus \mathcal{M}})
\]

Meaning:
- \(x_t\): masked token to predict
- \(x_{\setminus \mathcal{M}}\): remaining visible tokens
- bidirectional context helps representation learning for understanding tasks

### 27.2 Decoder-style autoregressive modeling (GPT-like)
The objective is

\[
\mathcal{L}_{\text{AR}}
=
-\sum_{t=1}^{T}\log p(x_t \mid x_{<t})
\]

Meaning:
- predict token \(x_t\) from all previous tokens
- ideal for open-ended generation

---

## 28. Why scaling matters

A major discovery in the LLM era is that performance changes predictably with model size, data size, and compute budget.

A simplified view of scaling laws is

\[
L(N,D,C) \approx aN^{-\alpha} + bD^{-\beta} + cC^{-\gamma} + \epsilon
\]

Interpretation:
- \(L\): loss
- \(N\): number of parameters
- \(D\): amount of training data
- \(C\): compute
- \(a,b,c,\alpha,\beta,\gamma\): fitted constants
- as scale grows, loss often decreases according to approximate power laws

The practical lesson is not just “make the model bigger.”  
Compute-optimal scaling showed that training a too-large model on too-few tokens can be suboptimal. Balanced scaling matters.

---

## 29. Emergent capabilities and why they are debated

Researchers observed that larger models can show qualitatively new behaviors:
- in-context learning
- instruction following
- chain-of-thought sensitivity
- stronger coding ability
- transfer without gradient updates

However, “emergence” is debated because:
- some behaviors may appear abrupt only due to metric thresholds
- evaluation choices can exaggerate discontinuity
- capabilities may be gradual in underlying probability space

So the careful stance is:
- larger models often show qualitatively stronger behavior,
- but the exact notion of emergence should be treated critically.

---

## 30. In-context learning

In-context learning means the model can adapt behavior from examples inside the prompt, without changing its parameters.

Prompt example:
- task description
- a few input-output demonstrations
- a final query

The model then computes

\[
p(y \mid x, \text{prompt examples})
\]

This was one of the major practical breakthroughs highlighted by GPT-3.

---

## 31. Instruction tuning and alignment

Raw pretrained models are not necessarily helpful, truthful, or safe. Post-training aims to align them better with human preferences.

### 31.1 Supervised instruction tuning
Train on prompt-response pairs:

\[
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T}\log p(y_t \mid y_{<t}, x)
\]

Meaning:
- \(x\): prompt/instruction
- \(y_t\): response token
- this is standard teacher-forced fine-tuning on demonstrations

### 31.2 Reward modeling
A reward model \(r_\psi(x,y)\) is trained from ranked human preferences.

A common pairwise objective is

\[
\mathcal{L}_{\text{RM}}
=
-\log \sigma\left(r_\psi(x,y^+) - r_\psi(x,y^-)\right)
\]

Meaning:
- \(y^+\): preferred answer
- \(y^-\): less preferred answer
- \(r_\psi\): learned scalar reward
- if \(r_\psi(x,y^+)\) is larger, the loss decreases

### 31.3 RLHF intuition
The language model is further optimized to produce responses with higher reward, while staying close to the pretrained or supervised model to avoid drift.

A stylized RLHF objective is

\[
\max_\pi \ \mathbb{E}_{y \sim \pi(\cdot \mid x)}[r(x,y)] - \beta \, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
\]

Meaning:
- \(\pi\): current policy/model
- \(r(x,y)\): reward for response \(y\) to prompt \(x\)
- \(\pi_{\text{ref}}\): reference model
- \(\mathrm{KL}\): Kullback–Leibler divergence
- \(\beta\): penalty strength
- the first term encourages preferred outputs; the second prevents the model from drifting too far

This is the high-level logic behind RLHF.

---

## 32. Parameter-efficient fine-tuning and LoRA

Fine-tuning all parameters of a large model can be expensive.

LoRA assumes the weight update is approximately low-rank:

\[
\Delta W = BA
\]

where
- \(W \in \mathbb{R}^{d \times k}\): original weight matrix
- \(\Delta W\): trainable update
- \(B \in \mathbb{R}^{d \times r}\)
- \(A \in \mathbb{R}^{r \times k}\)
- \(r \ll \min(d,k)\): low rank

Then the adapted weight is

\[
W' = W + \Delta W = W + BA
\]

Meaning:
- instead of training every element of \(W\), train only low-rank factors \(B\) and \(A\)
- this greatly reduces trainable parameters and memory cost

LoRA has become one of the most practically important adaptation methods.

---

## 33. Retrieval-Augmented Generation (RAG)

A core limitation of parametric language models is that their internal knowledge can be outdated, hard to verify, or difficult to attribute. RAG augments generation with external retrieval.

A simplified form is

\[
p(y \mid x) = \sum_{d \in \mathcal{D}} p(d \mid x)\, p(y \mid x, d)
\]

Meaning:
- \(x\): input question/prompt
- \(\mathcal{D}\): candidate documents
- \(p(d \mid x)\): probability or relevance of document \(d\) for prompt \(x\)
- \(p(y \mid x,d)\): generation conditioned on both prompt and retrieved document

Interpretation:
1. retrieve relevant evidence
2. condition the generator on retrieved evidence
3. generate an answer grounded in external context

Benefits:
- fresher knowledge
- improved factuality
- evidence attribution
- easier updates without retraining the base model

---

## 34. Long-context modeling

As LLMs became more capable, users wanted models to reason over:
- books
- codebases
- long reports
- multi-document collections
- long conversations

The challenge is that standard attention cost grows roughly quadratically with sequence length:

\[
\text{cost} \propto T^2
\]

Meaning:
- \(T\): sequence length
- every token attends to every other token, so the attention matrix has size \(T \times T\)

This creates memory and compute challenges.

Common solutions:
- better positional encodings
- long-context fine-tuning
- retrieval and chunking
- sparse attention
- sliding-window attention
- recurrence/compression hybrids

---

## 35. Mixture-of-Experts (MoE)

MoE increases model capacity without activating all parameters for every token.

A simplified MoE layer is

\[
h' = \sum_{e=1}^{E} g_e(x)\, f_e(x)
\]

Meaning:
- \(E\): number of experts
- \(f_e(x)\): output of expert \(e\)
- \(g_e(x)\): routing weight for expert \(e\)
- only a small subset of experts is often activated for each token

Benefits:
- much larger total parameter count
- lower per-token compute than equivalently dense models
- better scaling efficiency in many settings

Challenges:
- routing instability
- load balancing
- systems complexity

---

## 36. LLM evaluation

Evaluation is much broader than perplexity.

### 36.1 Pretraining metric
Perplexity is related to token-level negative log-likelihood.

If average token loss is \(\ell\), perplexity is

\[
\mathrm{PPL} = \exp(\ell)
\]

Meaning:
- lower perplexity means the model assigns higher probability to the true next tokens
- but lower perplexity does not guarantee better safety, truthfulness, or reasoning

### 36.2 Downstream evaluation
- QA accuracy
- MMLU-like benchmarks
- code benchmarks
- long-context recall
- instruction-following evaluation
- human preference evaluation
- factuality/hallucination tests
- safety and refusal behavior
- calibration and uncertainty estimation

### 36.3 Why evaluation is hard
- benchmark contamination is possible
- prompt formatting can change scores
- high benchmark score may not reflect deployment reliability
- “reasoning” is difficult to measure robustly

---

## 37. Hallucination, grounding, and factuality

LLMs can produce fluent but false outputs. This happens because the model is optimized to produce likely continuations, not guaranteed truths.

Important mitigation strategies:
- retrieval grounding
- tool use
- citation-aware interfaces
- verifier models
- domain-restricted deployment
- self-consistency and structured checking
- post-hoc fact verification

A useful conceptual distinction:
- **parametric memory**: knowledge stored in model weights
- **non-parametric memory**: external documents, databases, tools, search results

RAG combines both.

---

## 38. Tool use and agents

LLMs are increasingly used as controllers that call tools:
- search
- calculators
- code interpreters
- APIs
- databases
- planning modules

A simplified loop is:
1. read state/context
2. generate action/tool call
3. observe result
4. continue reasoning

This moves LLMs from pure text generators toward decision-making systems.

Important caution:
- tool use can improve accuracy
- but it introduces systems-level failure modes:
  - bad tool selection
  - incorrect arguments
  - cascading errors
  - unsafe automation

---

## 39. Multimodal LLMs

The boundary between “multimodal learning” and “LLMs” is increasingly blurred.

Modern multimodal LLMs usually:
1. encode non-text inputs such as images/audio/video
2. project them into tokens or token-like embeddings
3. feed them into a language-model-style decoder
4. generate text, actions, or multimodal outputs

A generic interface is

\[
p(y \mid x_{\text{text}}, x_{\text{image}}, x_{\text{audio}}, \dots)
\]

This is conceptually powerful because text becomes the universal reasoning/output space, while other modalities are treated as context sources.

Representative systems include Flamingo, BLIP-2, PaLI, LLaVA, and Gemini-style multimodal models.

---

## 40. Current trends in LLM research

### 40.1 Better post-training
The field increasingly recognizes that pretraining alone is not enough; post-training strongly shapes usability.

### 40.2 Retrieval and grounding
Grounded generation is central for enterprise, scientific, legal, and medical deployment.

### 40.3 Long-context systems
Models increasingly handle much longer contexts, but long-context reasoning remains harder than long-context storage.

### 40.4 Mixture-of-Experts and efficient scaling
Scaling is now about architecture efficiency, not just dense parameter count.

### 40.5 Multimodal integration
Language models are becoming central reasoning engines over image, video, audio, and documents.

### 40.6 Smaller but stronger open models
The field has shifted from “largest model wins” to careful data curation, post-training, and efficient adaptation.

### 40.7 Safety, evaluation, and reliability
As deployment grows, evaluation of factuality, robustness, alignment, and misuse resistance matters more.

---

## 41. Core limitations of LLMs

1. **Hallucination**  
   Fluency can hide error.

2. **Context fragility**  
   Small prompt changes can alter outputs.

3. **Reasoning uncertainty**  
   Strong benchmark performance does not imply guaranteed logical reliability.

4. **Data dependence**  
   Biases and artifacts from training data persist.

5. **Opaque internal representations**  
   Mechanistic understanding remains incomplete.

6. **Expensive evaluation**  
   Human-centered evaluation is costly and slow.

7. **Safety and misuse risk**  
   Powerful models can be misused at scale.

---

## 42. A unifying view: multimodal learning and LLMs are converging

A useful modern perspective is:

- traditional multimodal learning focused on **fusing specialized modalities for a task**
- LLM research focused on **scalable language modeling and instruction-following**
- current frontier systems combine both:
  - multimodal encoders
  - shared token spaces
  - language-based reasoning
  - retrieval
  - long context
  - tool use
  - post-training alignment

In other words, many frontier systems are best understood as **multimodal, retrieval-aware, instruction-following Transformer systems** rather than as isolated “vision models” or “language models.”

---

## 43. How to study this field effectively

A strong learning path is:

### Stage 1. Mathematical foundations
- linear algebra
- probability
- optimization
- information theory basics

### Stage 2. Neural sequence modeling
- RNN/LSTM limitations
- attention
- Transformer blocks

### Stage 3. Classical multimodal learning
- CCA
- DCCA
- early/late fusion
- tensor fusion

### Stage 4. Large-scale representation learning
- contrastive learning
- CLIP/ALIGN
- masked modeling and autoregressive modeling

### Stage 5. LLMs
- autoregressive pretraining
- scaling laws
- instruction tuning
- RLHF / preference optimization
- LoRA and efficient fine-tuning
- retrieval and grounding

### Stage 6. Frontier systems
- multimodal LLMs
- long-context reasoning
- agentic tool use
- safety and evaluation

---

## 44. Suggested GitHub repository structure

```text
textbook/
├─ README.md
├─ 01_multimodal_learning/
│  ├─ 01_introduction.md
│  ├─ 02_classical_foundations_cca_dcca.md
│  ├─ 03_fusion_strategies.md
│  ├─ 04_multimodal_transformers.md
│  ├─ 05_contrastive_learning_clip_align.md
│  ├─ 06_large_multimodal_models.md
│  ├─ 07_missing_modality_and_robustness.md
│  └─ 08_open_problems.md
├─ 02_llm/
│  ├─ 01_introduction.md
│  ├─ 02_transformer_basics.md
│  ├─ 03_pretraining_objectives.md
│  ├─ 04_scaling_laws.md
│  ├─ 05_instruction_tuning_alignment.md
│  ├─ 06_lora_and_efficient_finetuning.md
│  ├─ 07_rag_and_grounding.md
│  ├─ 08_long_context_moe_agents.md
│  └─ 09_evaluation_safety_open_problems.md
└─ references.md
```

---

## 45. Minimal reading roadmap for a graduate-level course

### Multimodal learning essentials
1. Ngiam et al. (2011)
2. Andrew et al. (2013)
3. Zadeh et al. (2017)
4. Tsai et al. (2019)
5. Radford et al. (2021)
6. Jia et al. (2021)
7. Alayrac et al. (2022)
8. Li et al. (2022, 2023)
9. Chen et al. (2022)
10. Liu et al. (2023)
11. Wu et al. (2024)

### LLM essentials
1. Vaswani et al. (2017)
2. Devlin et al. (2018/2019)
3. Brown et al. (2020)
4. Lewis et al. (2020)
5. Ouyang et al. (2022)
6. Hoffmann et al. (2022)
7. Hu et al. (2021/2022)
8. Zhao et al. (2023)
9. Minaee et al. (2024)

---

# References

## Multimodal Learning
1. Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., & Ng, A. Y. (2011). Multimodal Deep Learning. ICML.
2. Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013). Deep Canonical Correlation Analysis. ICML.
3. Zadeh, A., Chen, M., Poria, S., Cambria, E., & Morency, L.-P. (2017). Tensor Fusion Network for Multimodal Sentiment Analysis. EMNLP.
4. Tsai, Y.-H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L.-P., & Salakhutdinov, R. (2019). Multimodal Transformer for Unaligned Multimodal Language Sequences. ACL.
5. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
6. Jia, C., Yang, Y., Xia, Y., et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. ICML.
7. Alayrac, J.-B., Donahue, J., Luc, P., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS.
8. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. ICML.
9. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML.
10. Chen, X., Wang, X., Changpinyo, S., et al. (2022). PaLI: A Jointly-Scaled Multilingual Language-Image Model. arXiv.
11. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. NeurIPS.
12. Gemini Team. (2024). Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv.
13. Wu, R., Wang, H., Chen, H.-T., & Carneiro, G. (2024). Deep Multimodal Learning with Missing Modality: A Survey. arXiv.

## Large Language Models
14. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS.
15. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
16. Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. NeurIPS.
17. Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
18. Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
19. Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. NeurIPS.
20. Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). Training Compute-Optimal Large Language Models. Chinchilla. arXiv.
21. Zhao, W. X., Zhou, K., Li, J., et al. (2023). A Survey of Large Language Models. arXiv.
22. Minaee, S., Mikolov, T., Nikzad, N., et al. (2024). Large Language Models: A Survey. arXiv.
23. Cai, W., Jiang, J., Wang, F., Tang, J., Kim, S., & Huang, J. (2024). A Survey on Mixture of Experts in Large Language Models. arXiv.
24. Ding, Y., et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens. arXiv.

---

## Final note for textbook authorship

If you plan to expand this into a true textbook, the next best step is to split this file into chapter files and then add:

- chapter-level figures
- notation summary tables
- “common mistakes” sections
- derivation appendices
- code notebooks for each chapter
- benchmark/dataset tables
- reading questions and exercises
- domain-specific case studies such as healthcare, manufacturing, and multimodal RAG

This current draft is designed to be a strong starting scaffold for a GitHub textbook.
