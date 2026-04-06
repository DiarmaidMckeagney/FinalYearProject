#NOTE: It is recommended that you only run the parts you need and comment out the rest as it takes a long time to run everything.

import os
import BETHDatasetLoader
from Isolation_Forest import IsolationForest
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance

def run_models():
    trainingDataset, trainingLabels, validationDataset, validationLabels, testingDataset, testingLabels = BETHDatasetLoader.get_datasets()
    # resetting the files for the default model
    if os.path.exists("BETH_IsoFor.csv"):
        os.remove("BETH_IsoFor.csv")

    if os.path.exists("BETH_OCSVM.csv"):
        os.remove("BETH_OCSVM.csv")

    if os.path.exists("BETH_RoCo.csv"):
        os.remove("BETH_RoCo.csv")
    # running default models
    print("\nRunning Isolation Forest")
    IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels,"BETH_IsoFor.csv",True,None)

    print("\nRunning OCSVM")
    OCSVM.run_ocsvm(trainingDataset, testingDataset, testingLabels,"BETH_OCSVM.csv",True,None)

    print("\nRunning Robust Covariance")
    RobustCovariance.run_robust_covariance(trainingDataset, testingDataset, testingLabels,"BETH_RoCo.csv",True,None)

    #running hyperparameter tuning
    run_hyperparameter_tuning(trainingDataset,validationDataset,validationLabels)

    #Best params
    Iso_For_Best_Param = [250,1,400]
    OCSVM_Best_Param = [0.4,0.25,"constant"]
    Ro_Co_Best_Param = [False,0.1,0.1]

    #running final models
    print("\nRunning Isolation Forest Hypertuned")
    IsolationForest.isolation_forest_hyperparametered(trainingDataset, testingDataset, testingLabels,Iso_For_Best_Param, None, False, True)

    print("\nRunning OCSVM Hypertuned")
    OCSVM.run_ocsvm_hyperparametered(trainingDataset, testingDataset, testingLabels,OCSVM_Best_Param, None, False, True)

    print("\nRunning Robust Covariance Hypertuned")
    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset, testingDataset, testingLabels,Ro_Co_Best_Param, None, False, True)


def run_hyperparameter_tuning(trainingDataset, validationDataset, validationLabels):
    # Reseting Storage Files
    if os.path.isfile("BETH_IsolationForest_General_Hyperparameter.csv"):
        os.remove("BETH_IsolationForest_General_Hyperparameter.csv")

    if os.path.isfile("BETH_OCSVM_General_Hyperparameter.csv"):
        os.remove("BETH_OCSVM_General_Hyperparameter.csv")

    if os.path.isfile("BETH_RobustCovariance_General_Hyperparameter.csv"):
        os.remove("BETH_RobustCovariance_General_Hyperparameter.csv")

    # params for IForest
    num_estimators = [100, 150, 200, 250, 300]
    num_features = [1, 2, 3, 5, 7]
    num_samples = [100, 150, 200, 300, 400]

    for i,num_estimator in enumerate(num_estimators): # running hyperparameter tuning on Iso Forest
        for j,num_feature in enumerate(num_features):
            for k,num_sample in enumerate(num_samples):
                config = [num_estimator,num_feature,num_sample]
                if i == 0 and j == 0 and k == 0: # this decides whether to print a header in the file or not
                    IsolationForest.isolation_forest_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_IsolationForest_General_Hyperparameter.csv", True, False)
                else:
                    IsolationForest.isolation_forest_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_IsolationForest_General_Hyperparameter.csv", False, False)

    # params for Robust Covariance
    assume_centred = [True, False]
    support_fraction = [0.1, 0.2, 0.3, 0.4, 0.5]
    contamination = [0.01, 0.02, 0.05, 0.1, 0.15]

    for i,centred in enumerate(assume_centred): # running hyperparameter tuning on Robust Cov
        for j,support_frac in enumerate(support_fraction):
            for k,contam in enumerate(contamination):
                config = [centred,support_frac,contam]
                if i == 0 and j == 0 and k == 0:# this decides whether to print a header in the file or not
                    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,validationDataset,validationLabels,config,"BETH_RobustCovariance_General_Hyperparameter.csv",True,False)
                else:
                    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,validationDataset,validationLabels,config,"BETH_RobustCovariance_General_Hyperparameter.csv",False,False)

    # params for OCSVM
    nu = [0.05,0.1,0.2,0.3,0.4]
    eta0 = [0.05,0.1,0.15,0.2,0.25]
    learning_rate = ["constant","invscaling","adaptive"]

    for i,nu_num in enumerate(nu): # running hyperparameter tuning on OCSVM
        for j,eta in enumerate(eta0):
            for k,lr in enumerate(learning_rate):
                config = [nu_num,eta,lr]
                if i == 0 and j == 0 and k == 0:# this decides whether to print a header in the file or not
                    OCSVM.run_ocsvm_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_OCSVM_General_Hyperparameter.csv", True, False)
                else:
                    OCSVM.run_ocsvm_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "BETH_OCSVM_General_Hyperparameter.csv", False, False)



if __name__ == "__main__":
    run_models()