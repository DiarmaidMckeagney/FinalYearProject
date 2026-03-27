from torch.utils.data import TensorDataset
from BETHDatasetLoader import get_datasets
import torch
class BETHDataset(TensorDataset):
    """
    Data collected from BETH (honeypots) and setup for unsupervised training and testing.
    """
    def __init__(self, split='train', subsample=0):
        trData,trLabels,valData,valLabels,tesData,tesLabels = get_datasets()
        if split == 'train':
            data = trData
            labels = trLabels
        elif split == 'val':
            data = valData
            labels = valLabels
        elif split == 'test':
            data = tesData
            labels = tesLabels
        else:
            raise Exception("Error: Invalid 'split' given")

        self.data = torch.as_tensor(data.values, dtype=torch.int64)
        self.labels = torch.as_tensor(labels, dtype=torch.int64)
        super().__init__(self.data, self.labels)