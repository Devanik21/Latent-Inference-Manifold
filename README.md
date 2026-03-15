# Latent Inference Manifold

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Latent-Inference-Manifold?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Latent-Inference-Manifold?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Exploring the geometry of learned representations — manifold structure, intrinsic dimensionality, and latent space topology in deep neural networks.

---

**Topics:** `arc-agi-2` · `inference-time` · `latent-space` · `machine-learning` · `meta-learning` · `multi-agent-reinforcement-learning` · `multi-agent-systems` · `neurons` · `program-synthesis`

## Overview

Latent Inference Manifold is a research project investigating the geometric structure of the latent
representations learned by deep neural networks. The central hypothesis — supported by a substantial
body of recent theoretical work — is that neural networks learn to map high-dimensional input data
onto low-dimensional manifolds embedded in the network's hidden state space, and that the geometry
of these manifolds encodes semantically meaningful structure that can be characterised, measured,
and deliberately shaped.

The project implements a suite of tools for manifold analysis in neural latent spaces: intrinsic
dimensionality estimation (PCA-based, TWO-NN estimator, correlation dimension), geodesic distance
computation between latent points (approximated via k-NN graphs), curvature estimation (sectional
curvature approximation), and topology analysis (persistent homology via Ripser) — all applied to
the activation spaces of trained neural networks on standard benchmarks.

A key application of these tools is the Manifold Hypothesis validation: empirically testing whether
image datasets (MNIST, CIFAR-10, CelebA) lie on low-dimensional manifolds by measuring the intrinsic
dimensionality of neural activations at each layer and tracking how this evolves through the
network depth. The results illuminate the representation learning process: manifold dimensionality
typically decreases through the network until the classification layer, reflecting progressive
abstraction from pixel-level statistics to semantic category boundaries.

---

## Motivation

Understanding what neural networks learn — at the level of geometric structure rather than input-output
behaviour — is a central open problem in deep learning interpretability. This project approaches that
problem through differential geometry: treating the learned representation as a Riemannian manifold
and measuring its geometric properties. This provides a mathematical language for questions like
'how does adversarial training change the representation geometry?' and 'why do two-layer MLPs
generalise better than one-layer ones for this data distribution?'

---

## Architecture

```
Trained Neural Network (any architecture)
        │
  Activation extraction at each layer
  (hook-based, no architectural modification)
        │
  ┌──────────────────────────────────────────────┐
  │  Manifold Analysis Pipeline:                 │
  │  ├── Intrinsic dimensionality (TWO-NN, PCA) │
  │  ├── k-NN graph construction (FAISS)        │
  │  ├── Geodesic distance estimation          │
  │  ├── Sectional curvature approximation     │
  │  └── Persistent homology (Ripser)          │
  └──────────────────────────────────────────────┘
        │
  Geometric statistics: mean, variance, histogram
        │
  Layer-wise manifold evolution visualisation
```

---

## Features

### Activation Extraction Hooks
Non-invasive activation extraction from any PyTorch model layer using forward hooks — no architectural modification required, compatible with ResNets, Transformers, MLPs, and CNNs.

### Intrinsic Dimensionality Estimation
Three estimators: PCA-based (explained variance ratio), TWO-NN (maximum likelihood intrinsic dimension estimator, Facco et al. 2017), and correlation dimension (box-counting) — with variance estimation and comparison.

### k-NN Graph for Geodesic Approximation
FAISS-accelerated k-nearest neighbour graph construction in the latent space, with Dijkstra's shortest path for geodesic distance approximation between point pairs.

### Sectional Curvature Estimation
Discrete approximation of sectional curvature at sample points using the metric tensor estimated from local neighbourhood structure — measuring whether the manifold is locally flat, positively, or negatively curved.

### Persistent Homology Analysis
Topological data analysis via Ripser: Vietoris-Rips persistence diagrams for the latent space point cloud, measuring the number of connected components, loops, and voids (Betti numbers β₀, β₁, β₂).

### Layer-Wise Manifold Evolution
Track intrinsic dimensionality, mean geodesic distance, and topological complexity through each layer of a deep network — visualising how the representation geometry evolves from input to classification layer.

### Adversarial Geometry Comparison
Compare manifold geometry between standard and adversarially trained models — measuring how adversarial training affects the curvature and intrinsic dimension of the decision boundary neighbourhood.

### Inter-Class Manifold Separation
Measure the geodesic distance between class-conditional manifolds in the latent space, providing a geometric explanation for classification difficulty.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **PyTorch** | Model activation extraction | Forward hooks for layer activation capture |
| **FAISS** | Fast k-NN search | GPU-accelerated approximate nearest neighbour for geodesic computation |
| **Ripser / gudhi** | Persistent homology | Topological data analysis of latent point clouds |
| **scikit-learn** | Dimensionality reduction | PCA, t-SNE, UMAP for 2D projection |
| **NumPy / SciPy** | Geometric computations | Covariance, eigendecomposition, graph algorithms |
| **Plotly / Matplotlib** | Visualisation | Persistence diagrams, layer-wise intrinsic dim plots |
| **pandas** | Results management | Layer-wise geometric statistics tables |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Latent-Inference-Manifold.git
cd Latent-Inference-Manifold
python -m venv venv && source venv/bin/activate
pip install torch torchvision faiss-cpu ripser scikit-learn numpy scipy matplotlib plotly pandas
# Optional GPU FAISS: pip install faiss-gpu
streamlit run app.py
```

---

## Usage

```bash
# Analyse ResNet-18 activations on CIFAR-10
python analyse_manifold.py --model resnet18 --dataset cifar10 --layer layer4

# Layer-wise dimensionality profile
python layer_profile.py --model resnet18 --dataset mnist --output profile.json

# Persistent homology of latent space
python topology.py --activations activations_layer4.npy --max_dim 2

# Compare standard vs adversarial model geometry
python compare_geometry.py --model_std std_model.pt --model_adv adv_model.pt
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `resnet18` | PyTorch model architecture for analysis |
| `DATASET` | `cifar10` | Dataset: mnist, cifar10, celeba |
| `LAYER_NAME` | `layer4` | Network layer to extract activations from |
| `N_SAMPLES` | `5000` | Number of samples for manifold analysis |
| `K_NEIGHBOURS` | `15` | k for k-NN graph construction |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Latent-Inference-Manifold/
├── README.md
├── requirements.txt
├── LAteNT.py
├── council.py
├── latent_dictionary.py
├── memory.py
└── ...
```

---

## Roadmap

- [ ] Riemannian geometry of the loss landscape: connecting weight space manifold structure to generalisation
- [ ] Neural collapse analysis: verifying the terminal phase learning geometry in deep classifiers
- [ ] Manifold-constrained optimisation: gradient descent that stays on the data manifold
- [ ] Cross-architecture manifold comparison: does ResNet vs ViT learn different geometric structure for the same task?
- [ ] Theoretical connection: empirical validation of the manifold dimension vs. sample complexity bounds

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

Manifold analysis requires substantial compute for large activation matrices — analysing 50,000 CIFAR-10 activations with persistent homology can take 10–60 minutes. Use N_SAMPLES=2000–5000 for exploratory work and full datasets only for final quantitative results. FAISS GPU installation requires CUDA and significantly accelerates k-NN computation.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
