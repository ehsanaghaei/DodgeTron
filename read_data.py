import json
import os
import datetime
from functions import func_read_json, func_write_json
from paths import PATH_infostealer, PATH_data


def list_files(dir_=PATH_infostealer):
    families = [os.path.join(dir_, f) for f in os.listdir(dir_)]
    fnames = list()
    for family in families:
        fnames.extend([os.path.join(family, f) for f in os.listdir(family)])
    return families, fnames


def read_files(fnames):
    from collections import defaultdict
    malwares = defaultdict(dict)
    c_err = 0
    for f in fnames:
        print(f)
        data = func_read_json(f)
        fname = f.split("/")[-2]
        id = f.split("/")[-1].replace(".json", "")
        if data.get("behavior"):
            process_count = len(data["behavior"]["processes"])
            for i in range(process_count):
                if data["behavior"]["processes"][i]['calls']:
                    call = data["behavior"]["processes"][i]['calls']
                    malwares[fname][id] = call
        else:
            c_err += 1
            print(c_err, "*ERROR: No behavior key!")
            print(f"{c_err}- There is no 'behavior' key in \n{f}")
    return malwares


# def summarize_representations():


def build_representation(malwares, segments: list, joined=False):
    if segments is None:
        segments = ['api']
    from collections import defaultdict
    representations = defaultdict(dict)
    for family in malwares:
        for id_ in malwares[family]:

            sequence = list()
            calls = malwares[family][id_]
            for call, sub_call in enumerate(calls):
                # sub_call = calls[call]
                seq = ""
                for seg in segments:
                    seq += f"{seg}:{sub_call[seg]} "
                sequence.append(seq)
            if joined:
                sequence = " ".join(sequence)
            representations[family][id_] = sequence
    return representations


def build_representation_batch(fnames, segments: list, joined=False, save=True, load=True):
    if load:
        print("Load preprocessd representations")
        if type(load) == bool:
            fname = "/media/ea/SSD2/Projects/DodgeTron/data/representations-2023-04-19.json"
        else:
            fname = load
        return func_read_json(fname)
    else:
        if segments is None:
            segments = ['api']
        from collections import defaultdict
        representations = defaultdict(dict)

        c_err = 0
        for f_idx, f in enumerate(fnames):

            print(f"{f_idx}/{len(fnames)} -> {f}")
            data = func_read_json(f)
            family = f.split("/")[-2]
            id_ = f.split("/")[-1].replace(".json", "")
            if data.get("behavior"):
                process_count = len(data["behavior"]["processes"])
                for i in range(process_count):
                    if data["behavior"]["processes"][i]['calls']:
                        calls = data["behavior"]["processes"][i]['calls']
                        sequence = list()
                        for call, sub_call in enumerate(calls):
                            # sub_call = calls[call]
                            seq = ""
                            for seg in segments:
                                seq += f"{seg}:{sub_call[seg]} "
                            sequence.append(seq)
                        if joined:
                            sequence = " ".join(sequence)
                        representations[family][id_] = sequence

            else:
                c_err += 1
                print(c_err, "*ERROR: No behavior key!")
                print(f"{c_err}- There is no 'behavior' key in \n{f}")
        if save:
            func_write_json(representations, os.path.join(PATH_data, '-'.join(['representations', str(datetime.date.today())]) + ".json"))

    return representations


def z_algorithm(lst):
    n = len(lst)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and lst[z[i]] == lst[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z


def find_repeated_patterns_(lst, n):
    patterns = set()
    for i in range(len(lst) - n + 1):
        sublst = lst[i:i + n]
        z = z_algorithm(sublst)
        for j in range(len(z)):
            if z[j] > 0 and j + z[j] == len(z):
                pattern = tuple(sublst[j:j + z[j]])
                patterns.add(pattern)
    return patterns


def rep_summarizer(representations, shorten_duplicates=True, shorten_all=False):
    if shorten_duplicates:
        for family in representations:
            for id_ in representations[family]:
                call = representations[family][id_]
                count = 1

                prev_text = call[0]
                output = []

                for i in range(1, len(call)):
                    if call[i] == prev_text:
                        count += 1
                    else:
                        if count > 1:
                            output.append(f'{prev_text} x {count}')
                        else:
                            output.append(prev_text)
                        prev_text = call[i]
                        count = 1

                # output last text
                if count > 1:
                    output.append(f'{prev_text} x {count}')
                else:
                    output.append(prev_text)

                representations[family][id_] = output
    return representations


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

# if __name__ == "__main__":
#     families, fnames = list_files(PATH_infostealer)
#     malwares = read_files(fnames[100:110]+fnames[1000:1010]+fnames[2000:2010])
#     representations = build_representation(malwares, segments=['api'])
#     representations_sum = rep_summarizer(representations)
