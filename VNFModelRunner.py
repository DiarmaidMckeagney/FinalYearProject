import VNFDatasetLoader
from Isolation_Forest import IsolationForest
import os
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance

def run_models():
    trainingDataset, trainingLabels, validationDataset, validationLabels, testingDataset, testingLabels = VNFDatasetLoader.import_training_and_testing_data()

    trainingDataset = VNFDatasetLoader.run_label_encoding(trainingDataset)
    validationDataset = VNFDatasetLoader.run_label_encoding(validationDataset)
    testingDataset = VNFDatasetLoader.run_label_encoding(testingDataset)

    #run_individual_feature_selection(trainingDataset, validationDataset, validationLabels)

    # SRC MAC is the last item on this list and IP Protocol is second last. I have not included it so that the models have 1 feature left.
    featuresSortedByImportance = ["Dst IP","Dst Port","Dst data bytes","Bytes","Src Bytes","Src Port","Src IP","Packets","Protocols","Session Length","Src data bytes","Session Segments","Data bytes","Dst Bytes"]

    #run_reduction_feature_selection(trainingDataset,validationDataset,validationLabels,featuresSortedByImportance)

    featuresToDelete = ["Dst IP","Dst Port","Dst data bytes","Bytes","Src Bytes","Src Port","Src IP"]

    trainingDataset.drop(featuresToDelete,axis=1,inplace=True)
    validationDataset.drop(featuresToDelete,axis=1,inplace=True)
    testingDataset.drop(featuresToDelete,axis=1,inplace=True)

    print("running Isolation Forest with default values")
    IsolationForest.run_isolation_forest(trainingDataset,testingDataset,testingLabels,"VNF_Iso_For_Default.csv",True,None)
    print("running OCSVM with default values")
    OCSVM.run_ocsvm(trainingDataset,testingDataset,testingLabels,"VNF_OCSVM_Default.csv",True,None)
    print("running Robust Covariance with default values")
    RobustCovariance.run_robust_covariance(trainingDataset,testingDataset,testingLabels,"VNF_Ro_Co_Default.csv",True,None)


    #run_hyperparameter_tuning(trainingDataset, validationDataset, validationLabels)

    isoForest_Best_params = [300,5,1000]
    ocsvm_best_params = [0.05,0.1,"constant"]
    robustCo_best_params = [False,0.1,0.1]



    print("\nRunning Isolation Forest Tuned")
    IsolationForest.isolation_forest_hyperparametered(trainingDataset, testingDataset, testingLabels, isoForest_Best_params, None, False, True)


    print("\nRunning OCSVM Tuned")
    OCSVM.run_ocsvm_hyperparametered(trainingDataset, testingDataset, testingLabels, ocsvm_best_params, None, False, True)


    print("\nRunning Robust Covariance Tuned")
    RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,testingDataset,testingLabels,robustCo_best_params,None,False,True)

def run_individual_feature_selection(trainingDataset, validationDataset, validationLabels):
    if os.path.isfile("IsolationForest_Feature_Selection.csv"):
        os.remove("IsolationForest_Feature_Selection.csv") #clear previous runs

    if os.path.isfile("OCSVM_Feature_Selection.csv"):
        os.remove("OCSVM_Feature_Selection.csv")

    if os.path.isfile("RobustCovariance_Feature_Selection.csv"):
        os.remove("RobustCovariance_Feature_Selection.csv")

    print("Running Isolation Forest with full feature set")
    IsolationForest.run_isolation_forest(trainingDataset, validationDataset, validationLabels, "IsolationForest_Feature_Selection.csv", True, "None")

    print("\nRunning OCSVM with full feature set")
    OCSVM.run_ocsvm(trainingDataset, validationDataset, validationLabels, "OCSVM_Feature_Selection.csv", True, "None")

    print("\nRunning Robust Covariance with full feature set")
    RobustCovariance.run_robust_covariance(trainingDataset, validationDataset, validationLabels, "RobustCovariance_Feature_Selection.csv", True, "None")

    for feature in trainingDataset:
        featurelessTrainingDataset = trainingDataset.copy()
        featurelessValidationDataset = validationDataset.copy()
        featurelessTrainingDataset.drop(feature, axis=1, inplace=True)
        featurelessValidationDataset.drop(feature, axis=1, inplace=True)
        print("\nRunning Isolation Forest without: ", feature)
        IsolationForest.run_isolation_forest(featurelessTrainingDataset, featurelessValidationDataset, validationLabels, "IsolationForest_Feature_Selection.csv", False, feature)

        print("\nRunning OCSVM without: ", feature)
        OCSVM.run_ocsvm(featurelessTrainingDataset, featurelessValidationDataset, validationLabels, "OCSVM_Feature_Selection.csv", False, feature)

        print("\nRunning Robust Covariance without: ", feature)
        RobustCovariance.run_robust_covariance(featurelessTrainingDataset, featurelessValidationDataset, validationLabels, "RobustCovariance_Feature_Selection.csv", False, feature)

def run_reduction_feature_selection(trainingDataset, validationDataset, validationLabels, importanceList):
    if os.path.isfile("IsolationForest_Feature_Selection_Reduction.csv"):
        os.remove("IsolationForest_Feature_Selection_Reduction.csv")

    if os.path.isfile("OCSVM_Feature_Selection_Reduction.csv"):
        os.remove("OCSVM_Feature_Selection_Reduction.csv")

    if os.path.isfile("RobustCovariance_Feature_Selection_Reduction.csv"):
        os.remove("RobustCovariance_Feature_Selection_Reduction.csv")

    reducedTrainingDataset = trainingDataset.copy()
    reducedValidationDataset = validationDataset.copy()

    print("Running Isolation Forest with full feature set")
    IsolationForest.run_isolation_forest(reducedTrainingDataset, reducedValidationDataset, validationLabels, "IsolationForest_Feature_Selection_Reduction.csv", True, 0)

    print("\nRunning OCSVM with full feature set")
    OCSVM.run_ocsvm(reducedTrainingDataset, reducedValidationDataset, validationLabels, "OCSVM_Feature_Selection_Reduction.csv", True, 0)

    print("\nRunning Robust Covariance with full feature set")
    RobustCovariance.run_robust_covariance(reducedTrainingDataset, reducedValidationDataset, validationLabels, "RobustCovariance_Feature_Selection_Reduction.csv", True, 0)

    for i,feature in enumerate(importanceList):
        reducedTrainingDataset.drop(feature, axis=1, inplace=True)
        reducedValidationDataset.drop(feature, axis=1, inplace=True)
        print(f"\nRunning Isolation Forest without {i+1} features. Dropped {feature}.")
        IsolationForest.run_isolation_forest(reducedTrainingDataset, reducedValidationDataset, validationLabels, "IsolationForest_Feature_Selection_Reduction.csv", False, i + 1)

        print(f"\nRunning OCSVM without: {i+1} features. Dropped {feature}.")
        OCSVM.run_ocsvm(reducedTrainingDataset, reducedValidationDataset, validationLabels, "OCSVM_Feature_Selection_Reduction.csv", False, i + 1)

        print(f"\nRunning Robust Covariance without: {i+1} features. Dropped {feature}.")
        RobustCovariance.run_robust_covariance(reducedTrainingDataset, reducedValidationDataset, validationLabels, "RobustCovariance_Feature_Selection_Reduction.csv", False, i + 1)

    #OCSVM and Robust Covariance Fails with only two features but these don't so I will manually call them here with the last one
    reducedTrainingDataset.drop("IP Protocol", axis=1, inplace=True)
    reducedValidationDataset.drop("IP Protocol", axis=1, inplace=True)
    print("\nRunning Isolation Forest without 15 features. Dropped IP Protocol.")
    IsolationForest.run_isolation_forest(reducedTrainingDataset, reducedValidationDataset, validationLabels, "IsolationForest_Feature_Selection_Reduction.csv", False, 15)

def run_hyperparameter_tuning(trainingDataset, validationDataset, validationLabels):
    if os.path.isfile("IsolationForest_General_Hyperparameter.csv"):
        os.remove("IsolationForest_General_Hyperparameter.csv")
    #
    # if os.path.isfile("OCSVM_General_Hyperparameter.csv"):
    #     os.remove("OCSVM_General_Hyperparameter.csv")

    # if os.path.isfile("RobustCovariance_General_Hyperparameter.csv"):
    #     os.remove("RobustCovariance_General_Hyperparameter.csv")

    num_estimators = [100,150,200,250,300]
    num_features = [1,2,3,5,7]
    num_samples = [100,150,200,300,400]

    for i,num_estimator in enumerate(num_estimators):
        for j,num_feature in enumerate(num_features):
            for k,num_sample in enumerate(num_samples):
                config = [num_estimator,num_feature,num_sample]
                if i == 0 and j == 0 and k == 0:
                    IsolationForest.isolation_forest_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "IsolationForest_General_Hyperparameter.csv", True, False)
                else:
                    IsolationForest.isolation_forest_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "IsolationForest_General_Hyperparameter.csv", False, False)

    assume_centred = [True, False]
    support_fraction = [0.1,0.2,0.3,0.4,0.5]
    contamination = [0.01,0.02,0.05,0.1,0.15]

    # for i,centred in enumerate(assume_centred):
    #     for j,support_frac in enumerate(support_fraction):
    #         for k,contam in enumerate(contamination):
    #             config = [centred,support_frac,contam]
    #             if i == 0 and j == 0 and k == 0:
    #                 RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,validationDataset,validationLabels,config,"RobustCovariance_General_Hyperparameter.csv",True,False)
    #             else:
    #                 RobustCovariance.run_robust_covariance_hyperparametered(trainingDataset,validationDataset,validationLabels,config,"RobustCovariance_General_Hyperparameter.csv",False,False)

    nu = [0.05,0.1,0.2,0.3,0.4]
    eta0 = [0.05,0.1,0.15,0.2,0.25]
    learning_rate = ["constant","invscaling","adaptive"]

    # for i,nu_num in enumerate(nu):
    #     for j,eta in enumerate(eta0):
    #         for k,lr in enumerate(learning_rate):
    #             config = [nu_num,eta,lr]
    #             if i == 0 and j == 0 and k == 0:
    #                 OCSVM.run_ocsvm_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "OCSVM_General_Hyperparameter.csv", True, False)
    #             else:
    #                 OCSVM.run_ocsvm_hyperparametered(trainingDataset, validationDataset, validationLabels, config, "OCSVM_General_Hyperparameter.csv", False, False)
    #


if __name__ == "__main__":
    run_models()

