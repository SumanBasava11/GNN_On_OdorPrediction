# Molecular Odor Prediction with Graph Neural Networks

## Overview

Quantitative Structure-Odor Relationship (QSOR) modeling seeks to predict how a molecule smells based on its structure. With advancements in deep learning and graph neural networks (GNNs) that can learn directly from molecular graphs [1–4], this long-standing interdisciplinary challenge—spanning chemistry, neuroscience, and machine learning—is gaining renewed attention [5].

Human odor perception is driven by interactions between odorant molecules and hundreds of olfactory receptors in the nose [6, 7]. By leveraging machine learning to model this process, we can design novel synthetic fragrances, reduce dependency on natural raw materials, and improve our understanding of how the brain processes smells.

This project builds upon recent work to improve predictive modeling for molecular odor using GNNs.

## Objectives

- **Improve Baseline Results**  
  Reproduce and enhance results from the paper *Machine Learning for Scent* [8].

- **Incorporate Functional Group Features**  
  Extend molecular feature representations by adding functional group information.

- **Add Odor Hierarchy**  
  Enrich odor space using hierarchical relationships between odor descriptors  
  _e.g., `meaty → chicken → roast chicken`_.

- **Include Explainability**  
  Introduce subgraph-based explanation methods to highlight molecular regions influencing to specific odors.

---

### Setup
```sh
# Clone the repository
git clone https://github.com/SumanBasava11/GNN_On_OdorPrediction.git
cd GNN_On_OdorPrediction

# Create a virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```sh
# Train and Evaluate the model
python main2.py
```
## References

1. Gilmer et al. (2017). *Neural Message Passing for Quantum Chemistry*.  
2. Wu et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning*.  
3. Duvenaud et al. (2015). *Convolutional Networks on Graphs for Learning Molecular Fingerprints*.  
4. Kipf & Welling (2016). *Semi-Supervised Classification with Graph Convolutional Networks*.  
5. Amoore & Pfaffmann (1969). *A Plan to Identify Odor Qualities of Molecules*.  
6. Buck & Axel (1991). *A Novel Multigene Family May Encode Odorant Receptors*.  
7. Mainland et al. (2014). *The Missense of Smell: Functional Variability in the Human Odorant Receptor Repertoire*.  
8. Sánchez-Lengeling et al. (2023). *Machine Learning for Scent: Learning Generalizable Perceptual Representations of Small Molecules*.

## License
This project is licensed under the MIT License.

