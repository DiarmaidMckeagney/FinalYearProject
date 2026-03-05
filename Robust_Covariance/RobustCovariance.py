import numpy as np
from sklearn.covariance import EllipticEnvelope
import Evaluation

def run_robust_covariance(trainingDataset, testingDataset, testingLabels,isBeth):

    robustCovariance = EllipticEnvelope(contamination=0.05,support_fraction= 0.6,random_state=87248935) # create model
    robustCovariance.fit(trainingDataset) # train the model

    print(np.unique(testingLabels, return_counts=True))

    predictions = robustCovariance.predict(testingDataset)

    Evaluation.evaluate_model(testingLabels, predictions,isBeth)