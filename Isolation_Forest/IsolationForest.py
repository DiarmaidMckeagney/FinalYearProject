import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

import Evaluation
import VNFDatasetLoader

if __name__ == "__main__":
    trainingData, trainingLabels, testingDataset, testingLabels = VNFDatasetLoader.import_training_and_testing_data()

    for col in trainingData.select_dtypes(include=['object']).columns: # use label encoding to encode any non-numeric column
        labelEncoder = LabelEncoder()
        trainingData[col] = labelEncoder.fit_transform(trainingData[col].astype(str))

    isolationForest = IsolationForest(n_estimators=200, max_features=6, contamination=0.05, random_state=380) # create model
    isolationForest.fit(trainingData) # train model

    for col in testingDataset.select_dtypes(include=['object']).columns: # perform label encoding on the testing data.
        labelEncoder = LabelEncoder()
        testingDataset[col] = labelEncoder.fit_transform(testingDataset[col].astype(str))

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingDataset) # test the model
    Evaluation.evaluate_model(testingLabels, predictions)
