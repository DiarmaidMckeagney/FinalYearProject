import VNFDatasetLoader
from Isolation_Forest import IsolationForest
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance

def run_models():
    trainingDataset, trainingLabels, testingDataset, testingLabels = VNFDatasetLoader.import_training_and_testing_data()

    trainingDataset = VNFDatasetLoader.run_label_encoding(trainingDataset)
    testingDataset = VNFDatasetLoader.run_label_encoding(testingDataset)

    IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels)
    OCSVM.run_ocsvm(trainingDataset, testingDataset, testingLabels)
    RobustCovariance.run_robust_covariance(trainingDataset, testingDataset, testingLabels)


if __name__ == "__main__":
    run_models()

