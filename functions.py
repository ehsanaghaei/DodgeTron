def func_read_json(fname):
    print(f"[func_read_json()] -> read {fname}")
    import json
    with open(fname, 'r') as f:
        content = json.load(f)
    return content


def func_write_json(dic, fname):
    print(f"[func_write_json()] -> write {fname}")
    import json
    # write the data to the file
    with open(fname, 'w') as file:
        json.dump(dic, file)
