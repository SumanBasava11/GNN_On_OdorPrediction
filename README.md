# GNN_On_OdorPrediction

## Overview
Olfactometry is the study of odor properties of a product (such as food) based on its molecular composition. Each product is decomposed into a set of odorant molecules, whose odors are identified by a human judge. This project aims to discover and explain the discriminating molecular features that correspond to specific odors using **Graph Neural Networks (GNNs)**.

By representing each molecule as a labeled graph with odor descriptors, I apply **GNN-based learning methods** to extract frequent sub-graphs characteristic of each odor. These sub-graphs correspond to the odorant parts of molecules, following a key/lock model.

## Methodology
1. **Dataset Preparation:**
   - Collect and preprocess odorant molecule datasets.
   - Label molecules with corresponding odor descriptors.
2. **Model Selection & Training:**
   - Implement GCN, GAT, GAE, and GCL models.
   - Train models using odor-labeled graphs.
3. **Feature Extraction & Explanation:**
   - Extract discriminative molecular substructures.
   - Apply contrastive learning and chemical function augmentation.
4. **Evaluation & Comparison:**
   - Compare against baseline methods.
   - Validate results with existing molecular odor databases.

## Installation
### Prerequisites
- Python 3.8+
- PyTorch
- DGL (Deep Graph Library)
- RDKit (for molecular structure handling)
- Scikit-learn
- NumPy, Pandas, Matplotlib

### Setup
```sh
# Clone the repository
git clone https://github.com/SumanBasava11/GNN_on_Odor.git
cd GNN_on_Odor

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```sh
# Train the model
python train.py --config config.yaml

# Evaluate the model
python evaluate.py --checkpoint model.pth
```

## References
1. Jaubert, J.-N., Tapiero, C., & Dore, J.-C. (1995). The Field of Odors: Toward a Universal Language for Odor Relationships. *Perfumer & Flavorist, 20*, 1-16.
2. Sanchez-Lengeling, B., Wei, J. N., Lee, B. K., Gerkin, R. C., Aspuru-Guzik, A., & Wiltschko, A. B. (2019). Machine learning for scent: Learning generalizable perceptual representations of small molecules. *arXiv preprint arXiv:1910.10685*.
3. Wiltschko, A. B. (2019). Learning to Smell: Using Deep Learning to Predict the Olfactory Properties of Molecules. *Google Research Blog*.
4. Sisson, L. (2023). Olfactory Label Prediction on aroma-chemical Pairs. *arXiv preprint arXiv:2312.16124*. [https://arxiv.org/html/2312.16124v2](https://arxiv.org/html/2312.16124v2)
5. Ren, Y., Liu, B., Huang, C., Dai, P., Bo, L., & Zhang, J. (2019). Heterogeneous deep graph infomax. *arXiv preprint arXiv:1911.08538*. [https://arxiv.org/pdf/1911.08538](https://arxiv.org/pdf/1911.08538)
6. Ying, R., Wang, A., You, J., & Leskovec, J. (2020). Frequent subgraph mining by walking in order embedding space. *Proc. Int. Conf. Mach. Learn. Workshops*. [http://snap.stanford.edu/frequent-subgraph-mining/](http://snap.stanford.edu/frequent-subgraph-mining/)
7. Huang, Q., Yamada, M., Tian, Y., Singh, D., & Chang, Y. (2022). Graphlime: Local interpretable model explanations for graph neural networks. *IEEE Transactions on Knowledge and Data Engineering*.
8. Yuan, H., Yu, H., Gui, S., & Ji, S. (2022). Explainability in graph neural networks: A taxonomic survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(5), 5782-5799.
9. Barsainyan, A. A. et al. (2023). openPOM. [https://github.com/BioMachineLearning/openpom](https://github.com/BioMachineLearning/openpom)

## License
This project is licensed under the MIT License.

