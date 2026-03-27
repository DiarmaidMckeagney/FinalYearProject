import numpy as np
from sklearn.covariance import EllipticEnvelope
import Evaluation

def run_robust_covariance(trainingDataset, testingDataset, testingLabels, FileToWriteTo, isStart, feature):

    robustCovariance = EllipticEnvelope(random_state=556) # create model
    robustCovariance.fit(trainingDataset) # train the model

    print(np.unique(testingLabels, return_counts=True))

    predictions = robustCovariance.predict(testingDataset)

    Evaluation.evaluate_model(testingLabels, predictions, FileToWriteTo, isStart, feature)