import numpy as np
from sklearn.linear_model import SGDOneClassSVM

import Evaluation

def run_ocsvm(trainingDataset, testingDataset, testingLabels, FileToWriteTo, isStart, feature):

    ocsvm = SGDOneClassSVM(max_iter=3_000,random_state=556) # creating the OCSVM
    ocsvm.fit(trainingDataset) # training the model

    print(np.unique(testingLabels, return_counts=True))

    predictions = ocsvm.predict(testingDataset)

    Evaluation.evaluate_feature_selection_model(testingLabels, predictions, FileToWriteTo, isStart, feature)

def run_ocsvm_hyperparametered(trainingDataset, testingDataset, testingLabels, configList, fileToWriteTo, isStart, isFinal):
    ocsvm = SGDOneClassSVM(nu=configList[0],eta0=configList[1],learning_rate=configList[2],max_iter=3_000, random_state=556)  # creating the OCSVM
    ocsvm.fit(trainingDataset)  # training the model

    predictions = ocsvm.predict(testingDataset)
    if not isFinal:
        Evaluation.evaluate_hyper_model(testingLabels, predictions, isStart, fileToWriteTo, configList)
    else:
        Evaluation.final_eval_model(testingLabels, predictions)