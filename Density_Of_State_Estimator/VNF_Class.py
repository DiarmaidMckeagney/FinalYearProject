import torch
from VNFDatasetLoader import import_training_and_testing_data, run_label_encoding
from torch.utils.data import TensorDataset

class VNFDataset(TensorDataset):
    def __init__(self, split="train"):
        trData, trLabels, tesData, tesLabels = import_training_and_testing_data()
        if split == 'train':
            data = trData
            labels = trLabels
        elif split == 'test':
            data = tesData
            labels = tesLabels
        else:
            raise Exception("Error: Invalid 'split' given")

        data = run_label_encoding(data)
        print(labels.ctypes)

        self.data = torch.as_tensor(data.values, dtype=torch.int64)
        self.labels = torch.as_tensor(labels, dtype=torch.int64)
        super().__init__(self.data, self.labels)