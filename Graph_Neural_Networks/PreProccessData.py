import numpy as np
from torch_geometric.data import TensorAttr
import torch
import math
import os
dir_path = "/home/diarmaid/Documents/VNF_Dataset"

def importData():
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            with open(csv_file, "r") as f:
               print(csv_file)


if __name__ == '__main__':
    importData()