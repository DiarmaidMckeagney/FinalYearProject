import numpy as np
from torch_geometric.data import TensorAttr
import torch
import math
from multiprocessing import Pool, cpu_count
import os


dir_path = "/home/diarmaid/Documents/VNF_Dataset" # TODO - Make this a dynamic import, maybe include the dataset in the project directory
tensor_list = []
ip_dict = {}


def getFilePaths():
    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths


def processFile(filepath):
    with open(filepath, "r") as f:
        arr = np.loadtxt(f, delimiter=",",skiprows=1)
        return torch.tensor(arr)


if __name__ == '__main__':
    paths = getFilePaths()

    print(f"Found {len(paths)} CSV files.")

    with Pool(cpu_count()) as pool:
        tensor_list = pool.map(processFile, paths)

    print(f"Loaded {len(tensor_list)} tensors.")