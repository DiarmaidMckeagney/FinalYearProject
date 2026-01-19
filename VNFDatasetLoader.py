import os
import random
import pandas as pd

def getFilePaths():
    # this function gets all the filepaths in the VNF_Dataset folder.
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"FinalYearProject/VNF_Dataset")  # might be a better solution to this but it works
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
    datasetLabels = fullDataset.iloc[:, fullDataset.shape[1] - 1].values # separate out the labels
    fullDataset.drop("Label", axis=1, inplace=True) # drop the labels from the dataset

    return fullDataset, datasetLabels # return dataset and labels

def addContamination(filesToUse, filesNotToUse, contaminationAmount):
    contamination = pd.DataFrame() # used to hold the contamination dataframe to be returned
    random.shuffle(filesToUse)

    for file in filesToUse:
        if contamination.shape[0] >= contaminationAmount:
            return contamination

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


    return contamination # If somehow we get to the end of the all the files and still don't have enough, then return what we have


if __name__ == "__main__": # this is just used to test the contamination function
    filepaths = getFilePaths()
    print(filepaths)
    training_files = getFirstSessions(filepaths)

    training_files.append(filepaths[2]) # this is also done in main. It excludes the testing dataset from the contamination. In the actual code the training datasets are imported first before this is called.
    training_files.append(filepaths[3])

    contaminate = addContamination(filepaths,training_files,60_000)
    print(contaminate)