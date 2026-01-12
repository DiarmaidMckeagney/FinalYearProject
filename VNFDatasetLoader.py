
import os
dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "FinalYearProject/VNF_Dataset")# might be a better solution to this but it works

def getFilePaths():
    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths






