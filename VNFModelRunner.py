import VNFDatasetLoader
from Isolation_Forest import IsolationForest
import os
from One_Class_Support_Vector_Machine import OCSVM
from Robust_Covariance import RobustCovariance

def run_models():
    trainingDataset, trainingLabels, testingDataset, testingLabels = VNFDatasetLoader.import_training_and_testing_data()

    trainingDataset = VNFDatasetLoader.run_label_encoding(trainingDataset)
    testingDataset = VNFDatasetLoader.run_label_encoding(testingDataset)

    #run_individual_feature_selection(trainingDataset, testingDataset, testingLabels)

    # SRC MAC is the last item on this list and IP Protocol is second last. I have not included it so that the models have 1 feature left.
    featuresSortedByImportance = ["Dst IP","Dst Port","Dst data bytes","Bytes","Src Bytes","Src Port","Src IP","Packets","Protocols","Session Length","Src data bytes","Session Segments","Data bytes","Dst Bytes"]

    run_reduction_feature_selection(trainingDataset,testingDataset,testingLabels,featuresSortedByImportance)

    featuresToDelete = ["Dst IP","Dst Port","Dst data bytes","Bytes","Src Bytes","Src Port","Src IP"]

    trainingDataset.drop(featuresToDelete,axis=1,inplace=True)
    testingDataset.drop(featuresToDelete,axis=1,inplace=True)




def run_individual_feature_selection(trainingDataset, testingDataset, testingLabels):
    if os.path.isfile("IsolationForest_Feature_Selection.csv"):
        os.remove("IsolationForest_Feature_Selection.csv") #clear previous runs

    if os.path.isfile("OCSVM_Feature_Selection.csv"):
        os.remove("OCSVM_Feature_Selection.csv")

    if os.path.isfile("RobustCovariance_Feature_Selection.csv"):
        os.remove("RobustCovariance_Feature_Selection.csv")

    print("Running Isolation Forest with full feature set")
    IsolationForest.run_isolation_forest(trainingDataset, testingDataset, testingLabels,"IsolationForest_Feature_Selection.csv",True,"None")

    print("\nRunning OCSVM with full feature set")
    OCSVM.run_ocsvm(trainingDataset, testingDataset, testingLabels,"OCSVM_Feature_Selection.csv",True,"None")

    print("\nRunning Robust Covariance with full feature set")
    RobustCovariance.run_robust_covariance(trainingDataset, testingDataset, testingLabels,"RobustCovariance_Feature_Selection.csv",True,"None")

    for feature in trainingDataset:
        featurelessTrainingDataset = trainingDataset.copy()
        featurelessTestingDataset = testingDataset.copy()
        featurelessTrainingDataset.drop(feature, axis=1, inplace=True)
        featurelessTestingDataset.drop(feature, axis=1, inplace=True)
        print("\nRunning Isolation Forest without: ", feature)
        IsolationForest.run_isolation_forest(featurelessTrainingDataset, featurelessTestingDataset, testingLabels,"IsolationForest_Feature_Selection.csv",False,feature)

        print("\nRunning OCSVM without: ", feature)
        OCSVM.run_ocsvm(featurelessTrainingDataset, featurelessTestingDataset, testingLabels,"OCSVM_Feature_Selection.csv",False,feature)

        print("\nRunning Robust Covariance without: ", feature)
        RobustCovariance.run_robust_covariance(featurelessTrainingDataset, featurelessTestingDataset, testingLabels,"RobustCovariance_Feature_Selection.csv",False,feature)

def run_reduction_feature_selection(trainingDataset, testingDataset, testingLabels,importanceList):
    if os.path.isfile("IsolationForest_Feature_Selection_Reduction.csv"):
        os.remove("IsolationForest_Feature_Selection_Reduction.csv")

    if os.path.isfile("OCSVM_Feature_Selection_Reduction.csv"):
        os.remove("OCSVM_Feature_Selection_Reduction.csv")

    if os.path.isfile("RobustCovariance_Feature_Selection_Reduction.csv"):
        os.remove("RobustCovariance_Feature_Selection_Reduction.csv")

    reducedTrainingDataset = trainingDataset.copy()
    reducedTestingDataset = testingDataset.copy()

    print("Running Isolation Forest with full feature set")
    IsolationForest.run_isolation_forest(reducedTrainingDataset, reducedTestingDataset, testingLabels,"IsolationForest_Feature_Selection_Reduction.csv", True, 0)

    print("\nRunning OCSVM with full feature set")
    OCSVM.run_ocsvm(reducedTrainingDataset, reducedTestingDataset, testingLabels, "OCSVM_Feature_Selection_Reduction.csv", True, 0)

    print("\nRunning Robust Covariance with full feature set")
    RobustCovariance.run_robust_covariance(reducedTrainingDataset, reducedTestingDataset, testingLabels,"RobustCovariance_Feature_Selection_Reduction.csv", True, 0)

    for i,feature in enumerate(importanceList):
        reducedTrainingDataset.drop(feature, axis=1, inplace=True)
        reducedTestingDataset.drop(feature, axis=1, inplace=True)
        print(f"\nRunning Isolation Forest without {i+1} features. Dropped {feature}.")
        IsolationForest.run_isolation_forest(reducedTrainingDataset, reducedTestingDataset, testingLabels,"IsolationForest_Feature_Selection_Reduction.csv", False, i+1)

        print(f"\nRunning OCSVM without: {i+1} features. Dropped {feature}.")
        OCSVM.run_ocsvm(reducedTrainingDataset, reducedTestingDataset, testingLabels,"OCSVM_Feature_Selection_Reduction.csv", False, i+1)

        print(f"\nRunning Robust Covariance without: {i+1} features. Dropped {feature}.")
        RobustCovariance.run_robust_covariance(reducedTrainingDataset, reducedTestingDataset, testingLabels,"RobustCovariance_Feature_Selection_Reduction.csv", False, i+1)

    #OCSVM and Robust Covariance Fails with only two features but these don't so I will manually call them here with the last one
    reducedTrainingDataset.drop("IP Protocol", axis=1, inplace=True)
    reducedTestingDataset.drop("IP Protocol", axis=1, inplace=True)
    print("\nRunning Isolation Forest without 15 features. Dropped IP Protocol.")
    IsolationForest.run_isolation_forest(reducedTrainingDataset, reducedTestingDataset, testingLabels,"IsolationForest_Feature_Selection_Reduction.csv", False, 15)

def run_hyperparameter_tuning(trainingDataset, testingDataset, testingLabels):
    if os.path.isfile("IsolationForest_General_Hyperparameter.csv"):
        os.remove("IsolationForest_General_Hyperparameter.csv")

    if os.path.isfile("OCSVM_General_Hyperparameter.csv"):
        os.remove("OCSVM_General_Hyperparameter.csv")

    if os.path.isfile("RobustCovariance_General_Hyperparameter.csv"):
        os.remove("RobustCovariance_General_Hyperparameter.csv")

    num_estimators = [100,150,200,250,300]
    num_features = [1,3,5,7,9]
    num_samples = [10,50,100,150,200]

    for num_estimator in num_estimators:
        for num_feature in num_features:
            for num_sample in num_samples:
                config = [num_estimator,num_feature,num_sample]
                IsolationForest.isolation_forest_hyperparameter_tuning(trainingDataset,testingDataset,testingLabels,config,"IsolationForest_General_Hyperparameter.csv",False,i)

if __name__ == "__main__":
    run_models()

