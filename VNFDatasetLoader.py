import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_file_paths():
    # this function gets all the filepaths in the VNF_Dataset folder.
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"FinalYearProject/VNF_Dataset")  # might be a better solution to this but it works

    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        # this line gets the path to the "csv" folder that is in each VNF service directory.
        current_path = os.path.join(dir_path, i, "v" + i, "csv")

        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths


def get_first_sessions(filePaths):
    #this function is used to get the filepaths to first session from each VNF service. I am doing this because the first sessions have only benign data.
    trainingFiles = []

    for file in filePaths:
        # this block goes through each path and gets the session number of the file.
        file_path_array = file.split("/")
        file_name_array = file_path_array[len(file_path_array) - 1]
        file_name_array = file_name_array.split("_")
        session_number = file_name_array[1]

        # now it checks if the session number is 1 and if so, appends it to the trainingFiles list
        if int(session_number) == 1:
            trainingFiles.append(file)
    return trainingFiles


def import_dataset_from_files(filePaths):
    dataset = []

    for file in filePaths:
        datasetFrame = pd.read_csv(file, header=0, low_memory=False, encoding="utf-8", on_bad_lines="skip", skipinitialspace=True) #read in each file
        datasetFrame.dropna(axis=0, how='all', inplace=True) # This drops any rows that have no values
        dataset.append(datasetFrame) # add the dataframe to the array of all dataframes

    fullDataset = pd.concat(dataset) # this merges all the data from each file into one dataframe

    numberOfAnomaliesNeeded = round((float(fullDataset.shape[0]) / 95.0) * 5.0)

    datasetLabels = fullDataset.iloc[:, fullDataset.shape[1] - 1].values # separate out the labels
    fullDataset.drop("Label", axis=1, inplace=True) # drop the labels from the dataset

    return fullDataset, datasetLabels, numberOfAnomaliesNeeded # return dataset and labels

def add_contamination(filesToUse, filesNotToUse, contaminationAmount):
    contamination = pd.DataFrame() # used to hold the contamination dataframe to be returned
    random.shuffle(filesToUse) # shuffling the files to that different anomalies will be added.

    for file in filesToUse:
        if contamination.shape[0] >= contaminationAmount: # if we have enough
            contaminationLabels = contamination.iloc[:, contamination.shape[1] - 1].values  # separate out the labels
            contamination.drop("Label", axis=1, inplace=True)  # drop the labels from the dataset
            return contamination, contaminationLabels

        if file in filesNotToUse: # we are skipping any files used in the training and testing datasets
            continue

        datasetContamination = pd.read_csv(file, header=0, low_memory=False, encoding="utf-8",on_bad_lines="skip", skipinitialspace=True) # read in file
        datasetContamination.dropna(axis=0, how='all', inplace=True) # remove null rows
        datasetContamination.drop(datasetContamination[datasetContamination["Label"] == "Benign"].index, axis=0, inplace=True) # drop benign data

        amountStillNeeded = contaminationAmount - contamination.shape[0]# the amount of rows still needed to be collected

        if datasetContamination.shape[0] < amountStillNeeded: #if the numRows in this file is less then the amount of contamination rows we still need to get
            contamination = pd.concat([contamination,datasetContamination])# concat the whole remaining file
            continue

        contamination = pd.concat([contamination,datasetContamination.iloc[0:amountStillNeeded,:]]) # add the remaining amount of samples needed.

    # If somehow we get to the end of the all the files and still don't have enough, then return what we have
    contaminationLabels = contamination.iloc[:, contamination.shape[1] - 1].values  # separate out the labels
    contamination.drop("Label", axis=1, inplace=True)  # drop the labels from the dataset

    return contamination, contaminationLabels


def import_training_and_testing_data():
    files = get_file_paths()  # read in filepaths
    trainingFiles = get_first_sessions(files)  # get the first sessions to be used as the training data

    dataset, labels, numAnomalies = import_dataset_from_files(trainingFiles)  # import the dataset

    # these next two lines are added so that the testing data is not added to the training data.
    trainingFiles.append(files[2])
    trainingFiles.append(files[3])

    anomalyData, anomalyLabels = add_contamination(files.copy(), trainingFiles, numAnomalies)

    fullDataset = pd.concat([dataset, anomalyData])
    fullLabels = np.append(labels, anomalyLabels)

    fullDataset.dropna(axis=1, how="any", inplace=True)
    print(fullDataset.shape)

    testingDataset, testingLabels, testNumAnomalies = import_dataset_from_files([files[2], files[3]])  # load some test data
    testingDataset.dropna(axis=1, how="any", inplace=True)

    print(testingDataset.shape)

    return fullDataset, fullLabels, testingDataset, testingLabels


def run_label_encoding(dataset):
    for col in dataset.select_dtypes(include=['object']).columns: # use label encoding to encode any non-numeric column
        labelEncoder = LabelEncoder()
        dataset[col] = labelEncoder.fit_transform(dataset[col].astype(str))

    return dataset

if __name__ == "__main__": # this is just used to test the contamination function
    files = get_file_paths()
    print(files)
    trainingDataFiles = get_first_sessions(files)

    trainingDataFiles.append(files[2]) # this is also done in main. It excludes the testing dataset from the contamination. In the actual code the training datasets are imported first before this is called.
    trainingDataFiles.append(files[3])

    contaminate = add_contamination(files, trainingDataFiles, 60_000)
    print(contaminate)