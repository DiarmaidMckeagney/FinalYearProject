import os
import BETHDatasetLoader
from Isolation_Forest import IsolationForest
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance

def run_models():
    trainingDataset, trainingLabels, validationDataset, validationLabels, testingDataset, testingLabels = BETHDatasetLoader.get_datasets()
    if os.path.exists("BETH_IsoFor.csv"):
        os.remove("BETH_IsoFor.csv")

    if os.path.exists("BETH_OCSVM.csv"):
        os.remove("BETH_OCSVM.csv")

    if os.path.exists("BETH_RoCo.csv"):
        os.remove("BETH_RoCo.csv")

    print("\nRunning Isolation Forest")
    IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels,"BETH_IsoFor.csv",True,None)

    print("\nRunning OCSVM")
    OCSVM.run_ocsvm(trainingDataset, testingDataset, testingLabels,"BETH_OCSVM.csv",True,None)

    print("\nRunning Robust Covariance")
    RobustCovariance.run_robust_covariance(trainingDataset, testingDataset, testingLabels,"BETH_RoCo.csv",True,None)

    #run_hyperparameter_tuning(trainingDataset,validationDataset,validationLabels)
    Iso_For_Best_Param = [100,7,300]
    OCSVM_Best_Param = [0.3,0.1,"invscaling"]
    Ro_Co_Best_Param = [True,0.1,0.1]

    print("\nRunning Isolation Forest Hypertuned")
    IsolationForest.isolation_forest_hyperparametered(trainingDataset, testingDataset, testingLabels,Iso_For_Best_Param, None, False, True)

    print("\nRunning OCSVM Hypertuned")
    OCSVM.run_ocsvm_hyperparametered(trainingDataset, testingDataset, testingLabels,OCSVM_Best_Param, None, False, True)

    print("\nRunning Robust Covariance Hypertuned")
    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset, testingDataset, testingLabels,Ro_Co_Best_Param, None, False, True)


def run_hyperparameter_tuning(trainingDataset, validationDataset, validationLabels):
    if os.path.isfile("BETH_IsolationForest_General_Hyperparameter.csv"):
        os.remove("BETH_IsolationForest_General_Hyperparameter.csv")

    if os.path.isfile("BETH_OCSVM_General_Hyperparameter.csv"):
        os.remove("BETH_OCSVM_General_Hyperparameter.csv")

    if os.path.isfile("BETH_RobustCovariance_General_Hyperparameter.csv"):
        os.remove("BETH_RobustCovariance_General_Hyperparameter.csv")

    num_estimators = [100, 150, 200, 250, 300]
    num_features = [1, 2, 3, 5, 7]
    num_samples = [100, 150, 200, 300, 400]

    for i,num_estimator in enumerate(num_estimators):
        for j,num_feature in enumerate(num_features):
            for k,num_sample in enumerate(num_samples):
                config = [num_estimator,num_feature,num_sample]
                if i == 0 and j == 0 and k == 0:
                    IsolationForest.isolation_forest_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_IsolationForest_General_Hyperparameter.csv", True, False)
                else:
                    IsolationForest.isolation_forest_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_IsolationForest_General_Hyperparameter.csv", False, False)

    assume_centred = [True, False]
    support_fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    contamination = [0.01, 0.02, 0.05, 0.1, 0.15]

    for i,centred in enumerate(assume_centred):
        for j,support_frac in enumerate(support_fraction):
            for k,contam in enumerate(contamination):
                config = [centred,support_frac,contam]
                if i == 0 and j == 0 and k == 0:
                    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,validationDataset,validationLabels,config,"BETH_RobustCovariance_General_Hyperparameter.csv",True,False)
                else:
                    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,validationDataset,validationLabels,config,"BETH_RobustCovariance_General_Hyperparameter.csv",False,False)

    nu = [0.05,0.1,0.2,0.3,0.4]
    eta0 = [0.05,0.1,0.15,0.2,0.25]
    learning_rate = ["constant","invscaling","adaptive"]

    for i,nu_num in enumerate(nu):
        for j,eta in enumerate(eta0):
            for k,lr in enumerate(learning_rate):
                config = [nu_num,eta,lr]
                if i == 0 and j == 0 and k == 0:
                    OCSVM.run_ocsvm_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_OCSVM_General_Hyperparameter.csv", True, False)
                else:
                    OCSVM.run_ocsvm_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_OCSVM_General_Hyperparameter.csv", False, False)



if __name__ == "__main__":
    run_models()