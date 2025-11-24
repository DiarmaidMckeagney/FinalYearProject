from multiprocessing import Pool
from torch_geometric.data import HeteroData
import os
from collections import defaultdict
from typing import Dict
import pandas as pd

dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "VNF_Dataset")# might be a better solution to this but it works

def getFilePaths():
    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths



def extract_node_types(filepath):
    node_types : Dict[str,list] = defaultdict(list)
    chunksize = 10_000 # this chunk size could probably be bumped, I just wanted to be sure that there would be no memory issues with multiprocessing.
    index_col = 0 # sets the column number to use as the index (header)

    for chunk in pd.read_csv(filepath, chunksize=chunksize, index_col = index_col):
        chunk = chunk.reset_index(drop=False)
        orig_index_col = chunk.columns[0]  # reset_index placed original index first

        for hash_of_row, row in chunk.iterrows():
            # determine which columns are non-null for this row (exclude the row with the titles for all the columns)
            headerlessRow = row.drop(labels=[orig_index_col])
            notNullCols = tuple(col for col in headerlessRow.index if pd.notna(headerlessRow[col]))
            # stable node type name
            cols_sorted = tuple(sorted(notNullCols))
            for col in cols_sorted:
                if " " in col:
                    col.replace(" ", "-")
            node_name = "cols_" + "_".join(cols_sorted)
            # keep only the populated columns and preserve order = cols_sorted
            row_data = {col: headerlessRow[col] for col in cols_sorted}
            node_types[node_name].append(row_data)

    return node_types




# Example usage:
filepaths_to_use = getFilePaths()  # or your list of 32 CSV paths
nested_dict_for_multiprocessing = list

with Pool(8) as p:
    nested_dict_for_multiprocessing.append(p.map(extract_node_types, filepaths_to_use))

print(nested_dict_for_multiprocessing)





