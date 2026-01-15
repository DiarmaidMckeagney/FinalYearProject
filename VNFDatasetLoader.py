
import os
import pandas as pd
dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FinalYearProject/VNF_Dataset")# might be a better solution to this but it works

def getFilePaths():
    # this function gets all the filepaths in the VNF_Dataset folder.
    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths


def getFirstSessions(filepaths):
    #this function is used to get the filepaths to first session from each VNF service. I am doing this because the first sessions have only benign data.
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
        datasetFrame = pd.read_csv(file, header=0, low_memory=False, encoding="utf-8", on_bad_lines="skip", skipinitialspace=True) #read in each file
        datasetFrame.dropna(axis=0, how='all', inplace=True) # This drops any rows that have no values
        dataset.append(datasetFrame) # add the dataframe to the array of all dataframes

    fullDataset = pd.concat(dataset) # this merges all the data from each file into one dataframe
    fullDataset = fullDataset.dropna(axis=1)# this drops any column that contains null values. I am doing this to ensure only common columns are used in the model.
    print(fullDataset.shape)
    datasetLabels = fullDataset.iloc[:, fullDataset.shape[1] - 1].values # seperate out the labels
    fullDataset.drop("Label", axis=1, inplace=True) # drop the labels from the dataset

    return fullDataset, datasetLabels # return dataset and labels