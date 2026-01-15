
import os
import pandas as pd
dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FinalYearProject/VNF_Dataset")# might be a better solution to this but it works

def getFilePaths():
    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths


def getFirstSessions(filepaths):
    training_files = []
    for file in filepaths:
        file_path_array = file.split("/")
        file_name_array = file_path_array[len(file_path_array) - 1]
        file_name_array = file_name_array.split("_")
        session_number = file_name_array[1]
        if int(session_number) == 1:
            training_files.append(file)
    return training_files


def importDatasetFromFiles(filepaths):
    dataset = []
    for file in filepaths:
        datasetFrame = pd.read_csv(file, header=0, low_memory=False, encoding="utf-8", on_bad_lines="skip", skipinitialspace=True)
        datasetFrame.dropna(axis=0, how='all', inplace=True)
        dataset.append(datasetFrame)

    fullDataset = pd.concat(dataset)
    fullDataset = fullDataset.dropna(axis=1)
    print(fullDataset.shape)
    datasetLabels = fullDataset.iloc[:, fullDataset.shape[1] - 1].values
    fullDataset.drop("Label", axis=1, inplace=True)

    return fullDataset, datasetLabels