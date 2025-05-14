import numpy as np
from torch.utils.data import Sampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

class IterativeStratifiedBatchSampler(Sampler):
    """
    Creates batches where each batch approximately maintains the multilabel distribution
    using iterative stratification.
    """
    def __init__(self, labels, batch_size):
        self.labels = labels  # NumPy array of shape (num_samples, num_labels)
        self.batch_size = batch_size
        self.num_samples = labels.shape[0]
        self.indices = np.arange(self.num_samples)
        self.batches = self._create_batches()

    def _create_batches(self):
        # Use a StratifiedKFold strategy to split into batches
        n_splits = self.num_samples // self.batch_size
        stratifier = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        batches = []
        for _, batch_idx in stratifier.split(self.indices, self.labels):
            batches.append(batch_idx.tolist())
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
