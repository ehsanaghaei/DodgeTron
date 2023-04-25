import os

import torch

from functions import func_write_json, func_read_json
from paths import PATH_data


def handle_large_input_embedding(model, input_ids, window_size=512, overlap_size=128, cls_embedding=True, pooling=True):
    #    --> input_ids = tokenizer.encode(text, add_special_tokens=True)
    # Split input into chunks

    chunks = []

    for i in range(0, len(input_ids), window_size - overlap_size):
        start = max(0, i - overlap_size)
        end = min(len(input_ids), i + window_size)
        chunk = input_ids[start:end]
        chunks.append(chunk)
    print(f"[handle_large_input_embedding()] -> the input id size is {len(input_ids)}\n"
          f"chunk size: {len(chunks)}\n"
          f"---")

    # Process chunks with RoBERTa model
    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            input_ids = torch.tensor(chunk).unsqueeze(0)
            outputs = model(input_ids)
            embeddings.append(outputs[0][:, 0, :])  # Get <cls> token embeddings

    # with torch.no_grad():
    #     outputs = model(**chunks)

    # embeddings = outputs[0][:, 0, :]
    # Concatenate output embeddings
    if pooling:
        embedding = torch.cat(embeddings, dim=0)
        embedding = torch.mean(embedding, dim=0)
    else:
        embedding = torch.cat(embeddings, dim=0)
    return embedding


def handle_large_input_embedding_plus(model, tokenizer, input_ids, window_size=512, cls_embedding=True, pooling=True, save_fname=None):
    output_tensors = torch.split(input_ids, window_size)
    input_strings = [' '.join([str(x.item()) for x in tensor]) for tensor in output_tensors]
    print(f"[handle_large_input_embedding_plus()] -> No. Chunks: {len(output_tensors)}\n")

    # inputs = tokenizer.batch_encode_plus(input_strings, return_tensors='pt', padding=True, truncation=True)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #
    # embeddings = outputs.last_hidden_state[:, 0]  # Get <cls> token embeddings
    #
    # if pooling:
    #     embeddings = torch.mean(embeddings, dim=0)

    # ----
    fpath = os.path.join(PATH_data, "oversized_embeddings", save_fname)
    if not os.path.isfile(fpath) and save_fname:
        batch_size = 256
        all_outputs = list()
        for i in range(0, len(input_strings), batch_size):
            # Get the current batch of input data
            batch_inputs = tokenizer.batch_encode_plus(
                input_strings[i:min(i + batch_size, len(input_strings))],
                return_tensors='pt',
                padding=True,
                truncation=True
            )

            # Process the batch
            with torch.no_grad():
                batch_outputs = model(**batch_inputs)
                all_outputs.append(batch_outputs.last_hidden_state[:, 0])
            embeddings = torch.cat(all_outputs, dim=0)
            if pooling:
                embeddings = torch.mean(embeddings, dim=0)

        # ----

        if save_fname:

            print(f"[handle_large_input_embedding_plus()] -> save oversized embedding tensor: {save_fname}")

            torch.save(embeddings, fpath)
            return None
        else:
            return embeddings
    else:
        print(f"[handle_large_input_embedding_plus()] -> *Interrupted: {save_fname} exist.")
        return None


def get_embeddings(texts, model, tokenizer, cls_embedding=True, pooling=True):
    # does not handle the oversized input
    # model.to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**input_ids)
        if cls_embedding:
            embeddings = outputs[0][:, 0, :]
        else:
            embeddings = outputs[0]
            if pooling:
                embeddings = torch.mean(embeddings, dim=0)
    return embeddings


def get_malware_embeddings(representation, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=512, overlap_size=100, save=True):
    print(f"[get_malware_embeddings()]: -> identify oversized inputs")

    from collections import defaultdict
    rep_norm = defaultdict(dict)
    rep_oversize = defaultdict(dict)
    final_embedding = defaultdict(list)
    for family in representation:
        for id_ in representation[family]:
            text = " ".join(representation[family][id_])
            input_ids = tokenizer.encode(text, padding=True, truncation=False, return_tensors='pt')[0]
            text_size = len(input_ids)
            if text_size > max_size:
                rep_oversize[family][id_] = representation[family][id_]
                # embedding = handle_large_input_embedding(model, input_ids, window_size=max_size, overlap_size=overlap_size, cls_embedding=cls_embedding, pooling=mean_pooling)
            else:
                rep_norm[family][id_] = representation[family][id_]
                # embedding = get_embeddings(texts, model, tokenizer, cls_embedding=cls_embedding, pooling=mean_pooling)
    func_write_json(rep_norm, "./data/rep_norm.json")
    del rep_norm

    print(f"[get_malware_embeddings()]: -> get oversized embeddings")
    # get_oversized_embedding
    c1 = 0
    c2 = 0
    for family in rep_oversize:
        for id_ in rep_oversize[family]:
            print(f"Process {c1}/{len(rep_oversize)}:{c2}/{len(rep_oversize[family])}")
            text = " ".join(rep_oversize[family][id_])
            input_ids = tokenizer.encode(text, padding=True, truncation=False, return_tensors='pt')[0]
            if save:
                save_fname = f"{family}-{id_}"
            else:
                save_fname = False

            embedding = handle_large_input_embedding_plus(model, tokenizer, input_ids, window_size=max_size, cls_embedding=cls_embedding, pooling=mean_pooling, save_fname=save_fname)
            final_embedding[family].append(embedding)
            c2 += 1
        c1 += 1
        print(f"{c1}/{family}:\t samples number {c2}")

    print(f"[get_malware_embeddings()]: -> get normal embeddings")
    # get normal embeddings
    data = list()
    labels = list()
    rep_norm = func_read_json("./data/rep_norm.json")
    for family2 in rep_norm:
        for id2_ in rep_norm[family2]:
            data.append(" ".join(rep_norm[family2][id2_]))
            labels.append(family2)

    del rep_norm
    print(f"[get_malware_embeddings()]: -> retrieve embeddings")
    normal_embeddings = get_embeddings(data, model, tokenizer, cls_embedding=cls_embedding, pooling=mean_pooling)
    for i, embed in enumerate(normal_embeddings):
        final_embedding[labels[i]].append(embed)
    return final_embedding


def find_frequent_patterns(sequences, threshold=20):
    from prefixspan import PrefixSpan

    # Use PrefixSpan to find frequent patterns in each sequence
    for i, sequence in enumerate(sequences):
        ps = PrefixSpan(sequence)
        frequent_patterns = ps.frequent(threshold)  # set the minimum support threshold to 2
        print(f"Malware {i}: {frequent_patterns}")

# s = [['a', 'b', 'c', 'd', 'e', 'f'],
#      ['b', 'c', 'e', 'f'],
#      ['c', 'd', 'e', 'f'],
#      ['e', 'f']
#      ]
# find_frequent_patterns(s, threshold=3)
# if __name__ == "__main__":
#
#     rep_norm, rep_oversize = get_malware_embeddings(representations, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=500, overlap_size=100)
