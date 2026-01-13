import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import VNFDatasetLoader

if __name__ == "__main__":
    files = VNFDatasetLoader.getFilePaths()
    dataset = pd.read_csv(files[0], header=0,low_memory=False,encoding="utf-8",on_bad_lines="skip")

    dataset = dataset.dropna(axis=1)

    datasetLabels = dataset.iloc[:,dataset.shape[1]-1].values

    dataset.drop("Label",axis=1,inplace=True)

    for col in dataset.select_dtypes(include=['object']).columns:
        label_encoder = LabelEncoder()
        dataset[col] = label_encoder.fit_transform(dataset[col].astype(str))


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