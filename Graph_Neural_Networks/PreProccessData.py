import numpy as np
import torch
from multiprocessing import Pool, cpu_count, Lock
import os
import struct

dir_path = "/home/diarmaid/Documents/VNF_Dataset" # TODO - Make this a dynamic import, maybe include the dataset in the project directory
tensor_list = []

ip_dict = {}
country_code_dict = {}
protocol_dict = {}
mac_address_dict = {}

icmp_dict = {0:0, 8:8, 3:3, 11:11}
ASN_dict = {}
version_dict = {}

URI_dict = {}
alt_names_dict = {}
target_dict = {}

locks = [Lock(), Lock(), Lock(), Lock(), Lock(), Lock(), Lock(), Lock(), Lock(), Lock()]


def getFilePaths():
    filepaths = []
    for i in [x for x in os.listdir(dir_path)]:
        current_path = os.path.join(dir_path, i, "v" + i, "csv")
        for file in os.listdir(current_path):
            csv_file = os.path.join(current_path, file)
            filepaths.append(csv_file)

    return filepaths

def queryDict(dictionary, query, lockNum):
    if query == "":
        return 0
    dict_keys = dictionary.keys()

    if query.count(",") > 0:
        queries = query.split(",")
        query_string = ""

        for q in queries:
            if q in dict_keys:
                query_string += dictionary[q]
            else:
                locks[lockNum].acquire()
                dictionary[q] = len(dictionary) + 1
                locks[lockNum].release()
                query_string += dictionary[q]

        return float(query_string)


    if query in dict_keys:
        return dictionary[query]
    else:
        locks[lockNum].acquire()
        dictionary[query] = len(dictionary) + 1
        locks[lockNum].release()

    return dictionary[query]

def hex_to_float_struct(hex_str): # got from https://www.geeksforgeeks.org/python/convert-hex-string-to-float-in-python/
    hex_str = hex_str.strip().replace("0x", "")
    byte_array = bytes.fromhex(hex_str)
    float_num = struct.unpack('!f', byte_array)[0]
    return float_num



def processFile(filepath):
    with open(filepath, "r") as f:
        arr = np.loadtxt(f,
                         delimiter=",",
                         skiprows=1,
                         converters={2: lambda x: queryDict(ip_dict,x,0),
                                     3: lambda x: queryDict(country_code_dict,x,1),
                                     5: lambda x: queryDict(ip_dict, x,0),
                                     7: lambda x: queryDict(country_code_dict,x,1),
                                     9: lambda x: queryDict(protocol_dict,x,2),
                                     10: lambda x: queryDict(protocol_dict,x,2),
                                     17: lambda x: hex_to_float_struct(x),
                                     18: lambda x: hex_to_float_struct(x),
                                     28: lambda x: queryDict(mac_address_dict,x,3),
                                     29: lambda x: queryDict(icmp_dict,x,4),
                                     31: lambda x: queryDict(ASN_dict,x,5),
                                     32: lambda x: queryDict(version_dict,x,6),
                                     33: lambda x: queryDict(URI_dict,x,7),
                                     34: lambda x: queryDict(alt_names_dict,x,8),
                                     37: lambda x: queryDict(country_code_dict, x,1),
                                     38: lambda x: queryDict(alt_names_dict,x,8),
                                     39: lambda x: queryDict(target_dict, x,9)
                                     }
                         )
        return torch.tensor(arr)


if __name__ == '__main__':
    paths = getFilePaths()

    print(f"Found {len(paths)} CSV files.")

    with Pool(cpu_count()) as pool:
        tensor_list = pool.map(processFile, paths)

    print(f"Loaded {len(tensor_list)} tensors.")