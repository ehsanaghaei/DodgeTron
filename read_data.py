import os
import json
import os
from pathlib import Path
import time
import zipfile

from functions import func_read_json
from paths import PATH_infostealer


def list_files(dir_=PATH_infostealer):
    families = [os.path.join(dir_, f) for f in os.listdir(dir_)]
    for family in families:
        fnames = [os.path.join(family, f) for f in os.listdir(family)]
    return families, fnames

def json_differences(file1, file2):
    with open(file1) as f:
        data1 = json.load(f)

    # Load data from file2.json
    with open(file2) as f:
        data2 = json.load(f)

    # Find common keys and values
    common_keys = set(data1.keys()) & set(data2.keys())
    common_values = {k: data1[k] for k in common_keys if data1[k] == data2[k]}

    print("Common keys:", common_keys)
    print("Common values:", common_values)

# ---------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    families, fnames = list_files(PATH_infostealer)

    # def read_data(path):


    for f in fnames:
        data = func_read_json(f)
        process_count = len(data["behavior"]["processes"])
        for i in range(process_count):
            if data["behavior"]["processes"][i]['calls']:
                print(data["behavior"]["processes"][i]['calls'])
