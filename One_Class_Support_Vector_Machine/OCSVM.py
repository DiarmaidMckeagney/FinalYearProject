import numpy as np
from sklearn.linear_model import SGDOneClassSVM

import Evaluation

def run_ocsvm(trainingDataset, testingDataset, testingLabels, FileToWriteTo, isStart, feature):

    ocsvm = SGDOneClassSVM(max_iter=3_000,random_state=556) # creating the OCSVM
    ocsvm.fit(trainingDataset) # training the model

    print(np.unique(testingLabels, return_counts=True))

    predictions = ocsvm.predict(testingDataset)

    Evaluation.evaluate_model(testingLabels, predictions, FileToWriteTo, isStart, feature)