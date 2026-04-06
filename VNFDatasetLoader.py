import math
import os
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


def get_sessions(filePaths):
    #this function takes the filepaths and separates the sessions from each VNF service into different arrays.
    trainingFiles = [] #all session 1's
    contaminationFiles = []#all session 2's
    validationFiles = []#all session 3's
    finalTestFiles = []#all session 4's

    for file in filePaths:
        # this block goes through each path and gets the session number of the file.
        file_path_array = file.split("/")
        file_name_array = file_path_array[len(file_path_array) - 1]
        file_name_array = file_name_array.split("_")
        session_number = file_name_array[1]

        # now it checks if the session number is 1 and if so, appends it to the trainingFiles list
        if int(session_number) == 1:
            trainingFiles.append(file)
        elif int(session_number) == 2: # if 2, append to contamination list
            contaminationFiles.append(file)
        elif int(session_number) == 3: # if 3, append to validation list
            validationFiles.append(file)
        elif int(session_number) == 4 or int(session_number) == 5: # if 4 or 5, append to test list
            finalTestFiles.append(file)
    return trainingFiles, contaminationFiles, validationFiles, finalTestFiles


def import_dataset_from_files(filePaths):
    dataset = []

    for file in filePaths: #reads in each file
        datasetFrame = pd.read_csv(file, header=0, low_memory=False, encoding="utf-8", on_bad_lines="skip", skipinitialspace=True) #read into dataframe
        datasetFrame.dropna(axis=0, how='all', inplace=True) # This drops any rows that have no values
        dataset.append(datasetFrame) # add the dataframe to the array of all dataframes

    fullDataset = pd.concat(dataset) # this merges all the data from each file into one dataframe

    numberOfAnomaliesNeeded = round((float(fullDataset.shape[0]) / 95.0) * 5.0) #calculates the number of samples needing to be added to have 5% contamination
    fullDataset["Label"] = (fullDataset["Label"] != "Benign").astype(int) # converts the labels into 0 or 1 depending on type.
    datasetLabels = fullDataset["Label"].astype('int64').values  # separate out the labels
    fullDataset.drop("Label", axis=1, inplace=True) # drop the labels from the dataset

    return fullDataset, datasetLabels, numberOfAnomaliesNeeded # return dataset and labels



def add_contamination(filesToUse, contaminationAmount):
    contamination = pd.DataFrame() # used to hold the contamination dataframe to be returned
    perFileCount = math.floor(contaminationAmount / 5)# want an even amount of samples from each file.

    for file in filesToUse:
        datasetContamination = pd.read_csv(file, header=0, low_memory=False, encoding="utf-8",on_bad_lines="skip", skipinitialspace=True) # read in file
        datasetContamination.dropna(axis=0, how='all', inplace=True) # remove null rows
        datasetContamination.drop(datasetContamination[datasetContamination["Label"] == "Benign"].index, axis=0, inplace=True) # drop benign data

        contamination = pd.concat([contamination,datasetContamination.iloc[0:perFileCount,:]]) # add the amount of samples needed.

    contamination.iloc[:, contamination.shape[1] - 1] = (contamination.iloc[:, contamination.shape[1] - 1] != "Benign").astype("int64") #converting the labels to 0 or 1
    contaminationLabels = contamination.iloc[:, contamination.shape[1] - 1].astype("int64").values  # separate out the labels
    contamination.drop("Label", axis=1, inplace=True)  # drop the labels from the dataset

    return contamination, contaminationLabels


def import_training_and_testing_data():
    files = get_file_paths()  # read in filepaths
    trainingFiles, contamFiles, valFiles, testFiles = get_sessions(files)  # get the sessions to be used as the training/val/test data

    dataset, labels, numAnomalies = import_dataset_from_files(trainingFiles)  # import the datasets

    anomalyData, anomalyLabels = add_contamination(contamFiles, numAnomalies)#get contamination

    fullDataset = pd.concat([dataset, anomalyData]) #merge the contamination into the training dataset
    fullLabels = np.append(labels, anomalyLabels).astype("int64")#merge the contamination labels into the training labels
    fullDataset.dropna(axis=1, how="any", inplace=True)# drop columns that contain null values

    validationDataset, validationLabels, valNumAnomalies = import_dataset_from_files(valFiles)  # load validation data
    validationDataset.dropna(axis=1, how="any", inplace=True)# drop columns that contain null values

    testingDataset, testingLabels, testNumAnomalies = import_dataset_from_files(testFiles)# load testing data
    fill_values = { # I was running into issues with the quality of the testing datasets and was not able to resolve it manually.
        "Dst IP": "0.0.0.0", # these values will be used to fill in the missing values in these files. I believe it is only one or two rows per feature.
        "Src IP": "0.0.0.0",
        "Dst Port": 0,
        "Src Port": 0
    }
    testingDataset.fillna(value=fill_values, inplace=True)
    testingDataset.drop(["Unnamed: 41"], axis=1, inplace=True)# I tried to remove this manually but it kept coming back so I am just dropping it everytime instead.
    testingDataset.dropna(axis=1, how="any", inplace=True) # drop columns that contain null values

    return fullDataset, fullLabels, validationDataset, validationLabels, testingDataset, testingLabels


def run_label_encoding(dataset):
    for col in dataset.select_dtypes(include=['object']).columns: # use label encoding to encode any non-numeric column
        labelEncoder = LabelEncoder()
        dataset[col] = labelEncoder.fit_transform(dataset[col].astype(str)).astype("float64")

    #removing the start and stop time columns
    if "Start Time" in dataset.columns:
        dataset.drop(["Start Time"], axis=1, inplace=True)
    if "Stop Time" in dataset.columns:
        dataset.drop(["Stop Time"], axis=1, inplace=True)
    return dataset
