import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

import Evaluation
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths() # read in filepaths
    training_files = VNFDatasetLoader.getFirstSessions(files) # get the first sessions to be used as the training data

    dataset, labels, numAnomalies = VNFDatasetLoader.importDatasetFromFiles(training_files) # import the dataset
    # these next two lines are added so that the testing data is not added to the training data.
    training_files.append(files[2])
    training_files.append(files[3])

    anomalyData, anomalyLabels = VNFDatasetLoader.addContamination(files.copy(),training_files,numAnomalies)

    fullDataset = pd.concat([dataset,anomalyData])
    fullLabels = np.append(labels,anomalyLabels)

    fullDataset.dropna(axis=1,how="any", inplace=True)
    print(fullDataset.shape)
    for col in fullDataset.select_dtypes(include=['object']).columns: # use label encoding to encode any non-numeric column
        label_encoder = LabelEncoder()
        fullDataset[col] = label_encoder.fit_transform(fullDataset[col].astype(str))

    isolationForest = IsolationForest(n_estimators=200, max_features=6, contamination=0.05, random_state=380) # create model
    isolationForest.fit(fullDataset) # train model

    testingDataset,testingLabels, testNumAnomalies = VNFDatasetLoader.importDatasetFromFiles([files[2], files[3]]) # load some test data
    testingDataset.dropna(axis=1, how="any", inplace=True)
    print(testingDataset.shape)
    for col in testingDataset.select_dtypes(include=['object']).columns: # perform label encoding on the testing data.
        label_encoder = LabelEncoder()
        testingDataset[col] = label_encoder.fit_transform(testingDataset[col].astype(str))

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingDataset) # test the model
    Evaluation.evaluate_model(testingLabels,predictions)
