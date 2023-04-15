def func_read_json(fname):
    import json
    with open(fname) as f:
        content = json.load(f)
    return content
