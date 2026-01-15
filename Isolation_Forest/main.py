import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths()
    training_files = []
    for file in files:
        file_path_array = file.split("/")
        file_name_array = file_path_array[len(file_path_array)-1]
        file_name_array = file_name_array.split("_")
        session_number = file_name_array[1]
        if int(session_number) == 1:
            training_files.append(file)

    fullTrainingFrames = []
    for file in training_files:
        dataset = pd.read_csv(files[3], header=0,low_memory=False,encoding="utf-8",on_bad_lines="skip")
        dataset.dropna(axis=0,how='all',inplace=True)
        fullTrainingFrames.append(dataset)

    fullTrainingDataset = pd.concat(fullTrainingFrames)
    fullTrainingDataset = fullTrainingDataset.dropna(axis=1)

    datasetLabels = fullTrainingDataset.iloc[:,fullTrainingDataset.shape[1]-1].values
    fullTrainingDataset.drop("Label",axis=1,inplace=True)

    for col in fullTrainingDataset.select_dtypes(include=['object']).columns:
        label_encoder = LabelEncoder()
        fullTrainingDataset[col] = label_encoder.fit_transform(fullTrainingDataset[col].astype(str))

    isolationForest = IsolationForest(n_estimators=200, max_features=6, random_state=56)
    isolationForest.fit(fullTrainingDataset)
    isolationForest.fit(fullTrainingDataset)

    testingdataset = pd.read_csv(files[2], header=0,low_memory=False,encoding="utf-8",on_bad_lines="skip")
    testingdataset.dropna(axis=0,how='all',inplace=True)
    testingdataset = testingdataset.dropna(axis=1)
    testingLabels = testingdataset.iloc[:,testingdataset.shape[1]-1].values
    testingdataset.drop("Label",axis=1,inplace=True)

    for col in testingdataset.select_dtypes(include=['object']).columns:
        label_encoder = LabelEncoder()
        testingdataset[col] = label_encoder.fit_transform(testingdataset[col].astype(str))

    print(np.unique(testingLabels, return_counts=True))

    predictions = isolationForest.predict(testingdataset)

    print(roc_auc_score(testingLabels, predictions))
    print("number of anomalies detected: ", list(predictions).count(-1))
    print("number of 'Benign' predictions: ", list(predictions).count(1))

