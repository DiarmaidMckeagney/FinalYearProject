import BETHDatasetLoader
from Isolation_Forest import IsolationForest
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance

def run_models():
    trainingDataset, trainingLabels, validationDataset, validationLabels, testingDataset, testingLabels = BETHDatasetLoader.get_datasets()

    IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels)


if __name__ == "__main__":
    run_models()