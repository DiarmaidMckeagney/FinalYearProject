import csv

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths()
    dataset = []
    isHeader = True
    with open(files[0], newline='', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # skipping header
        for row in csv_reader:
            dataset.append(row)

    print(len(dataset[0]))

    featureArray = []
    for i in range(len(dataset[0])):
        featureArray.append(0)
    for row in dataset:
        for i in range(len(row)):
            if row[i] != "":
                featureArray[i] = featureArray[i] + 1

    print(featureArray)

    # encoder = OneHotEncoder(sparse_output=False)
    #
    # df = pd.DataFrame(filled_in_dataset, columns=filled_in_dataset[0])
    # categorical_columns = df.select_dtypes(include=['object']).columns.tolist()


    # df_pandas_encoded = pd.get_dummies(df, columns=['Src Country', 'Dst Country', 'Protocols', 'IP Protocol', 'Dst ASN', 'Version', 'URI', 'Host',  'Alt Name', 'GEO', ' Hostname'], drop_first=True)
    #
    # one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    #
    # x_train = filled_in_dataset[1,0:40]
    #
    # isolation_forest = IsolationForest(n_estimators=100, max_samples=1000, random_state=0)
    #
    # isolation_forest.fit(x_train)