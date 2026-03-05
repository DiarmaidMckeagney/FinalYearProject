import torch
from torch.utils.data import DataLoader

import BETHDatasetLoader
from BETHDatasetLoader import BETHDataset
from Isolation_Forest import IsolationForest
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance
from Density_Of_State_Estimator import DoseAndVae

def run_models():
    # trainingDataset, trainingLabels, validationDataset, validationLabels, testingDataset, testingLabels = BETHDatasetLoader.get_datasets()
    #
    # print("\nRunning Isolation Forest")
    # IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels)
    #
    # print("\nRunning OCSVM")
    # OCSVM.run_ocsvm(trainingDataset, testingDataset, testingLabels)
    #
    # print("\nRunning Robust Covariance")
    # RobustCovariance.run_robust_covariance(trainingDataset, testingDataset, testingLabels)

    print("running DOSE + VAE model")

    trainDataset = BETHDataset(split="train")
    testDataset = BETHDataset(split="test")

    DoseAndVae.run_model(trainDataset,testDataset)

if __name__ == "__main__":
    run_models()