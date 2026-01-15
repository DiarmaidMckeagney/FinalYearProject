import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths()
    training_files = VNFDatasetLoader.getFirstSessions(files)

    dataset, labels = VNFDatasetLoader.importDatasetFromFiles(training_files)

    for col in dataset.select_dtypes(include=['object']).columns:
        label_encoder = LabelEncoder()
        dataset[col] = label_encoder.fit_transform(dataset[col].astype(str))

    isolationForest = IsolationForest(n_estimators=200, max_features=6, random_state=56)
    isolationForest.fit(dataset)

    testingDataset,testingLabels = VNFDatasetLoader.importDatasetFromFiles([files[2], files[3]])


    for col in testingDataset.select_dtypes(include=['object']).columns:
        label_encoder = LabelEncoder()
        testingDataset[col] = label_encoder.fit_transform(testingDataset[col].astype(str))

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingDataset)

    print(roc_auc_score(testingLabels, predictions))
    print("number of anomalies detected: ", list(predictions).count(-1))
    print("number of 'Benign' predictions: ", list(predictions).count(1))

