import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM

import Evaluation
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.get_file_paths()
    trainingFiles = VNFDatasetLoader.get_first_sessions(files)

    dataset, labels, numAnomalies = VNFDatasetLoader.import_dataset_from_files(trainingFiles)  # import the dataset
    # these next two lines are added so that the testing data is not added to the training data.
    trainingFiles.append(files[2])
    trainingFiles.append(files[3])

    anomalyData, anomalyLabels = VNFDatasetLoader.add_contamination(files.copy(), trainingFiles, numAnomalies)

    fullDataset = pd.concat([dataset, anomalyData])
    fullLabels = np.append(labels, anomalyLabels)

    fullDataset.dropna(axis=1, how="any", inplace=True)
    print(fullDataset.shape)
    for col in fullDataset.select_dtypes(
            include=['object']).columns:  # use label encoding to encode any non-numeric column
        labelEncoder = LabelEncoder()
        fullDataset[col] = labelEncoder.fit_transform(fullDataset[col].astype(str))

    ocsvm = OneClassSVM(kernel='rbf', gamma='auto') # creating the OCSVM
    ocsvm.fit(fullDataset) # training the model

    testingDataset, testingLabels, testNumAnomalies = VNFDatasetLoader.import_dataset_from_files([files[2], files[3]])  # load some test data
    testingDataset.dropna(axis=1, how="any", inplace=True)
    print(testingDataset.shape)
    for col in testingDataset.select_dtypes(include=['object']).columns:  # perform label encoding on the testing data.
        labelEncoder = LabelEncoder()
        testingDataset[col] = labelEncoder.fit_transform(testingDataset[col].astype(str))

    print(np.unique(testingLabels, return_counts=True))

    predictions = ocsvm.predict(testingDataset)

    Evaluation.evaluate_model(testingLabels, predictions)