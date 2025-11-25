import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from Graph_Neural_Networks import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths()
    dataset = []
    isHeader = True
    with open(files[0], "rb") as f:
        for row in f:
            if isHeader:
                isHeader = False
                continue

            dataset.append([row])

    imputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="0")

    imputer.fit(dataset)
    filled_in_dataset = np.array(imputer.transform(dataset))

    encoder = OneHotEncoder(sparse_output=False)

    df = pd.DataFrame(filled_in_dataset, columns=filled_in_dataset[0])
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()


    df_pandas_encoded = pd.get_dummies(df, columns=['Src Country', 'Dst Country', 'Protocols', 'IP Protocol', 'Dst ASN', 'Version', 'URI', 'Host',  'Alt Name', 'GEO', ' Hostname'], drop_first=True)

    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    x_train = filled_in_dataset[1,0:40]

    isolation_forest = IsolationForest(n_estimators=100, max_samples=1000, random_state=0)

    isolation_forest.fit(x_train)