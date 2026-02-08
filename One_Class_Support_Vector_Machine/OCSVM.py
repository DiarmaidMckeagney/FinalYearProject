import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM

import Evaluation
import VNFDatasetLoader

def run_ocsvm(trainingDataset, testingDataset, testingLabels):

    ocsvm = OneClassSVM(kernel='rbf', gamma='auto') # creating the OCSVM
    ocsvm.fit(trainingDataset) # training the model

    print(np.unique(testingLabels, return_counts=True))

    predictions = ocsvm.predict(testingDataset)

    Evaluation.evaluate_model(testingLabels, predictions)