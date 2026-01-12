import csv
# from json.decoder import NaN

import numpy as np
import pandas
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths()
    dataset = pandas.read_csv(files[0], header=0,low_memory=False,encoding="utf-8",on_bad_lines="skip")

    dataset = dataset.dropna(axis=1)
    print(dataset.head())
    print(dataset.shape)

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