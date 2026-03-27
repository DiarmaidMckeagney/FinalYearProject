import numpy as np
from sklearn.ensemble import IsolationForest
import Evaluation

def run_isolation_forest(trainingDataset, testingDataset, testingLabels,fileToWriteTo,isStart,feature):

    isolationForest = IsolationForest(random_state=556)  # create model
    isolationForest.fit(trainingDataset)  # train model

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingDataset)  # test the model
    Evaluation.evaluate_model(testingLabels, predictions, fileToWriteTo,isStart, feature)

def isolation_forest_hyperparameter_tuning(trainingDataset, testingDataset, testingLabels,configList,fileToWriteTo,isStart,feature):
    isolationForest = IsolationForest(n_estimators=configList[0],max_samples=configList[1],max_features=configList[2],random_state=556)
    isolationForest.fit(trainingDataset)
    print(np.unique(testingLabels, return_counts=True))
    predictions = isolationForest.predict(testingDataset)
    Evaluation.evaluate_model(testingLabels, predictions, fileToWriteTo,isStart, feature)