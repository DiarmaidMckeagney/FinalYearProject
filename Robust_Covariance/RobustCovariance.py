import numpy as np
from sklearn.covariance import EllipticEnvelope
import Evaluation

def run_robust_covariance(trainingDataset, testingDataset, testingLabels, FileToWriteTo, isStart, feature):

    robustCovariance = EllipticEnvelope(random_state=556) # create model
    robustCovariance.fit(trainingDataset) # train the model

    print(np.unique(testingLabels, return_counts=True))# prints the number of each label type in the testing set

    predictions = robustCovariance.predict(testingDataset)# test the model

    Evaluation.evaluate_feature_selection_model(testingLabels, predictions, FileToWriteTo, isStart, feature)#evaluate the model

def run_robust_covariance_hyperparametered(trainingDataset, testingDataset, testingLabels, configList, FileToWriteTo, isStart, isFinal):
    robustCovariance = EllipticEnvelope(assume_centered=configList[0],support_fraction=configList[1],contamination=configList[2],random_state=556)
    robustCovariance.fit(trainingDataset)
    predictions = robustCovariance.predict(testingDataset)# test the model
    if not isFinal:
        Evaluation.evaluate_hyper_model(testingLabels, predictions, isStart, FileToWriteTo, configList)#evaluate the hyperparametered model
    else:
        Evaluation.final_eval_model(testingLabels, predictions)#evaluate the final model