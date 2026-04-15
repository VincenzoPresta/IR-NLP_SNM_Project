
# 📘 Influence Cascade Decoding (ICD)

## Overview

This project explores a novel text generation strategy for Large Language Models (LLMs) based on an **influence network over tokens**.

The core idea is to **modify the decoding process** of an autoregressive Transformer by incorporating structural information derived from token transitions observed during simulated generations. This results in a **logit boosting mechanism** guided by a graph-based representation of token influence.

The approach combines:

* A pretrained LLM (Mistral-7B-Instruct)
* A graph-based influence network built from Monte Carlo simulations
* A diffusion process to estimate token importance
* A modified decoding strategy integrating these signals

---

## 🧠 Key Idea

Instead of relying solely on the model’s learned probabilities, this method:

1. **Builds a graph of token transitions**
2. **Assigns weights using statistical and semantic signals**
3. **Propagates influence using Independent Cascade**
4. **Boosts logits during decoding using graph-based signals**

Final decoding step:

[
\text{logit}_{final}(v) = \text{logit}(v) + \lambda \cdot (w(u, v) + bias(v))
]

Where:

* ( w(u,v) ): edge weight in the influence graph
* ( bias(v) ): activation probability from diffusion
* ( \lambda ): strength of the influence

---

## ⚙️ Pipeline

The full pipeline consists of the following steps:

### 1. Simulation

* Generate multiple sequences (e.g., 150 runs × 200 tokens)
* Use different sampling strategies:

  * Top-k
  * Top-p (nucleus)
  * Temperature sampling

### 2. Influence Graph Construction

* Nodes = tokens
* Edges = observed transitions
* Directed graph

### 3. Edge Weighting

Combination of:

* Empirical conditional probability
* PPMI (semantic association)

[
w(u, v) = \alpha \cdot \hat{p}(v|u) + (1 - \alpha) \cdot \text{PPMI}(u, v)
]

### 4. Influence Diffusion

* Model: **Independent Cascade**
* Seeds = tokens in the prompt
* Output: activation probabilities for each token

### 5. Logit Boosting

* Modify logits at each decoding step
* Use top-100 candidates, sample from top-50

---

## 📊 Experimental Analysis

### Parameters Studied

* **α (alpha)** → controls graph weighting
* **λ (lambda)** → controls influence strength during decoding

### Key Findings

* **Alpha**

  * Low α → semantic, dense diffusion
  * High α → sparse, frequency-driven graph
  * Best trade-off: **α = 0.4**

* **Lambda**

  * Increasing λ:

    * ↓ Perplexity
    * ↑ Influence token usage
    * ↑ Risk of repetition
  * Best range: **λ ∈ [1.5, 2.5]**

---

## 📈 Results

### Quantitative Evaluation

| Metric           | Base | Boosted |
| ---------------- | ---- | ------- |
| Coherence        | 6.22 | 6.21    |
| Informativeness  | 5.35 | 6.23    |
| Factuality       | 7.27 | 7.24    |
| Entropy          | 4.32 | 4.19    |
| Semantic Novelty | 0.40 | 0.40    |

Key insights:

* **Informativeness improves significantly**
* **Coherence and factuality remain stable**
* Slight reduction in lexical diversity

---

## 🧪 Evaluation Setup

* 100 samples per setting
* A/B comparison: **baseline vs boosted**
* Judge model: LLaMA-3.1-8B

Metrics:

* Coherence
* Informativeness
* Factuality
* Entropy (lexical diversity)
* Semantic novelty (SBERT)

---

## 🚀 Optimization

A custom **LogitsProcessor** was implemented to integrate the method into HuggingFace generation.

| Method                | Time   |
| --------------------- | ------ |
| Manual decoding       | 3m 07s |
| LogitsProcessor (GPU) | 18.97s |

→ ~**10× speedup**

---

## 📂 Datasets

* **CommonsenseQA** → prompts
* **All Tokens Simulations (custom)** → influence graph construction

---

## ⚠️ Limitations

* High computational cost for graph construction
* Requires many simulations per prompt
* Not yet practical for real-time applications

---

## 🔮 Future Work

* Precompute reusable influence networks
* Dynamic λ adaptation during generation
* Integration with semantic similarity retrieval
* Human evaluation

---

## 📌 Conclusion

Influence Cascade Decoding demonstrates that:

* Graph-based token relationships can effectively guide generation
* Logit boosting improves informativeness without harming coherence
* There exists a trade-off between control and diversity

The method opens new directions in **controlled text generation** and **graph-guided decoding**.

---

## 🧑‍💻 Authors

* Ivan Prisco
* Vincenzo Presta
* Matteo Greco
---

## Reference

Full report available in the repository:

* `Relazione_IRNLP-SOCIAL_Finale.pdf`
