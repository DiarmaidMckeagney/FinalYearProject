import numpy as np
from sklearn.ensemble import IsolationForest
import Evaluation

def run_isolation_forest(trainingDataset, testingDataset, testingLabels, isBeth):

    isolationForest = IsolationForest(contamination=0.05,random_state=56)  # create model
    isolationForest.fit(trainingDataset)  # train model

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingDataset)  # test the model
    Evaluation.evaluate_model(testingLabels, predictions,isBeth)
