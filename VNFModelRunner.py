import VNFDatasetLoader
from VNFDatasetLoader import VNFDataset
from Isolation_Forest import IsolationForest
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance
from Density_Of_State_Estimator import DoseAndVae

def run_models():
    trainingDataset, trainingLabels, testingDataset, testingLabels = VNFDatasetLoader.import_training_and_testing_data()

    trainingDataset = VNFDatasetLoader.run_label_encoding(trainingDataset)
    testingDataset = VNFDatasetLoader.run_label_encoding(testingDataset)

    print("Running Isolation Forest")
    IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels)

    print("\nRunning OCSVM")
    OCSVM.run_ocsvm(trainingDataset, testingDataset, testingLabels)

    print("\nRunning Robust Covariance")
    RobustCovariance.run_robust_covariance(trainingDataset, testingDataset, testingLabels)

    print("\nRunning VAE + DOSE")

    trainDataset = VNFDataset("train")
    testDataset = VNFDataset("test")

    DoseAndVae.run_model(trainDataset,testDataset)

if __name__ == "__main__":
    run_models()

