import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
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
        print(session_number)
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

    pd.set_option('display.max_columns', None)
    print(fullTrainingDataset.head())
    print(np.unique(datasetLabels,return_counts=True))

    isolationForest = IsolationForest(n_estimators=100,random_state=0)
    isolationForest.fit(fullTrainingDataset)