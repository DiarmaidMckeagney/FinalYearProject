import numpy as np
from sklearn.ensemble import IsolationForest
import Evaluation

def run_isolation_forest(trainingDataset, testingDataset, testingLabels):

    isolationForest = IsolationForest(n_estimators=200, max_features=6, contamination=0.05, random_state=380)  # create model
    isolationForest.fit(trainingDataset)  # train model

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingDataset)  # test the model
    Evaluation.evaluate_model(testingLabels, predictions)
