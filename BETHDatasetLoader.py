import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_file_paths():
    # this function gets all the filepaths in the BETH_Dataset folder.
    dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"FinalYearProject/BETH_Dataset")  # might be a better solution to this but it works

    trainingPath = os.path.join(dir_path,"labelled_training_data.csv")
    validationPath = os.path.join(dir_path,"labelled_validation_data.csv")
    testingPath = os.path.join(dir_path,"labelled_testing_data.csv")

    return trainingPath, validationPath, testingPath

def import_dataset_from_files(filePaths):
    print("Importing dataset")
    trainingDataset = pd.read_csv(filePaths[0])
    validationDataset = pd.read_csv(filePaths[1])
    testingDataset = pd.read_csv(filePaths[2])

    return trainingDataset, validationDataset, testingDataset

def get_datasets():
    paths = []

    trainPath, valPath, testPath = get_file_paths()

    paths.append(trainPath)
    paths.append(valPath)
    paths.append(testPath)

    trainingDataset, validationDataset, testingDataset = import_dataset_from_files(paths)

    trainingLabels = extractLabels(trainingDataset)
    validationLabels = extractLabels(validationDataset)
    testingLabels = extractLabels(testingDataset)

    trainingDataset.drop(["sus","evil"], axis=1, inplace=True)
    validationDataset.drop(["sus","evil"], axis=1, inplace=True)
    testingDataset.drop(["sus","evil"], axis=1, inplace=True)

    trainingDataset = process_dataset_columns(trainingDataset)
    validationDataset = process_dataset_columns(validationDataset)
    testingDataset = process_dataset_columns(testingDataset)

    return trainingDataset, trainingLabels, validationDataset, validationLabels, testingDataset, testingLabels

def extractLabels(dataset):
    labels = []

    originalLabels = dataset.iloc[:, -2:].values # getting the last two columns of the dataset (the labels)

    for label in originalLabels:
        if label[0] == 0 and label[1] == 0:
            labels.append("Benign")
        elif label[0] == 1 and label[1] == 0:
            labels.append("Suspicious")
        elif label[0] == 1 and label[1] == 1:
            labels.append("Evil")

    return labels

def process_dataset_columns(dataset):
    # This line is performing an operation on the processId column.
    # It changes any value of greater than 2 as false (0) and the others as True(1)
    # This is done as processIds 0,1,2 are values from the OS and all others are randomly assigned.
    dataset.iloc[:, 1] = (dataset.iloc[:, 1] <= 2).astype(int)

    # This is for parentProcessId. Same justification as ProcessId.
    dataset.iloc[:, 3] = (dataset.iloc[:, 3] <= 2).astype(int)

    # This is for userId. OS activity generally assigned num less than 1000.
    dataset.iloc[:, 4] = (dataset.iloc[:, 4] <= 1000).astype(int)

    # This is for mountNamespace. The BETH papers says that 4026531840 is the most common value.
    # This value corresponds to the /mmt directory.
    dataset.iloc[:, 5] = (dataset.iloc[:, 5] == 4026531840).astype(int)

    # This is for the processName. I am label encoding it to see if it can provide good results
    labelEncoder = LabelEncoder()
    dataset.iloc[:, 6] = labelEncoder.fit_transform(dataset.iloc[:, 6])

    # This is for return value. I had to do the weird -0.01, 0.99 because otherwise it wouldn't put them in the correct bin.
    # Sorts the return values into negative, 0, and positive.
    dataset.iloc[:, 12] = pd.cut(dataset.iloc[:, 12],bins=[-float("inf"), -0.01, 0.99, float("inf")],labels=[-1, 0, 1]).astype(int)

    # Removing other columns that are not needed.
    dataset.drop(["hostName","eventName","stackAddresses", "args"], axis=1, inplace=True)

    pd.set_option('display.max_columns', None)

    return dataset