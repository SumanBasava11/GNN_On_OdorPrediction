import torch
from sklearn.preprocessing import StandardScaler

class MolFeatureNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, mol_tensor_list):
        all = torch.stack(mol_tensor_list).numpy()
        self.scaler.fit(all)

    def transform(self, feat):
        normed = self.scaler.transform(feat.unsqueeze(0).numpy())
        return torch.tensor(normed.squeeze(0), dtype=torch.float)

    def transform_batch(self, batch_tensor):
        return torch.tensor(self.scaler.transform(batch_tensor.numpy()), dtype=torch.float)


class NodeFeatureNormalizer:
    def __init__(self, continuous_indices):
        self.continuous_indices = continuous_indices
        self.scaler = StandardScaler()

    def fit(self, node_tensor_list):
        all = torch.cat(node_tensor_list, dim=0)
        self.scaler.fit(all[:, self.continuous_indices].numpy())

    def transform(self, node_tensor):
        node_copy = node_tensor.clone()
        cont = node_copy[:, self.continuous_indices]
        normed = self.scaler.transform(cont.numpy())
        node_copy[:, self.continuous_indices] = torch.tensor(normed, dtype=torch.float)
        return node_copy
