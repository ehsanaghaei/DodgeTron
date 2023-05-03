import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from functions import func_write_json, func_read_json
from paths import PATH_data
import numpy as np
from read_data import read_tensor_from_file


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


def handle_large_input_embedding_plus(model, tokenizer, input_ids, window_size=512, cls_embedding=True, pooling=True, save_fname=None, device='cpu'):
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
        batch_size = 128
        all_outputs = list()
        for i in range(0, len(input_strings), batch_size):
            # Get the current batch of input data
            batch_inputs = tokenizer.batch_encode_plus(
                input_strings[i:min(i + batch_size, len(input_strings))],
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            if device.type == "cuda":

                input_ids = batch_inputs['input_ids'].to('cuda')
                attention_masks = batch_inputs['attention_mask'].to('cuda')
                with torch.no_grad():
                    batch_outputs = model(input_ids, attention_mask=attention_masks)
                    all_outputs.append(batch_outputs.last_hidden_state[:, 0].detach().cpu())
            # Process the batch

            else:
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


def collate_fn(batch):
    input_ids = [encoding.ids for encoding in batch]
    attention_mask = [encoding.attention_mask for encoding in batch]
    token_type_ids = [encoding.type_ids for encoding in batch]

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)

    return input_ids, attention_mask, token_type_ids
def get_embeddings(texts, model, tokenizer, cls_embedding=True, pooling=True, device='cpu'):
    # does not handle the oversized input
    # Create data loader
    batch_size = 8  # You can adjust this value to optimize performance and memory usage

    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            cls_embeddings = last_hidden_states[:, 0, :]
            embeddings.append(cls_embeddings.cpu())

    embeddings = np.concatenate(embeddings, axis=0)
    print("Embedding extraction done!")



    # embeddings = torch.cat(cls_embeddings, dim=0)
    """
    input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**input_ids)
    if cls_embedding:
        embeddings = outputs[0][:, 0, :]
    else:
        embeddings = outputs[0]
        if pooling:
            embeddings = torch.mean(embeddings, dim=0)
    """
    return embeddings


def get_malware_embeddings(representation, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=512, overlap_size=100, save=True, load_representations=True, load_embeddings=False, device='cpu'):
    print(f"[get_malware_embeddings()]: -> identify oversized inputs")

    from collections import defaultdict
    final_embedding = defaultdict(list)
    if not load_representations:
        rep_norm = defaultdict(dict)
        rep_oversize = defaultdict(dict)
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
        func_write_json(rep_oversize, "./data/rep_oversize.json")
        del rep_norm
    else:
        rep_oversize = func_read_json("./data/rep_oversize.json")

    print(f"[get_malware_embeddings()]: -> get oversized embeddings")
    # get_oversized_embedding
    c1 = 0
    c2 = 0
    if not load_embeddings:
        for family in rep_oversize:
            family = family.lower()
            for id_ in rep_oversize[family]:
                print(f"Process {c1}/{len(rep_oversize)}:{c2}/{len(rep_oversize[family])}")
                text = " ".join(rep_oversize[family][id_])
                input_ids = tokenizer.encode(text, padding=True, truncation=False, return_tensors='pt')[0]
                if save:
                    save_fname = f"{family}-{id_}"
                else:
                    save_fname = False

                embedding = handle_large_input_embedding_plus(model, tokenizer, input_ids, window_size=max_size, cls_embedding=cls_embedding, pooling=mean_pooling, save_fname=save_fname, device=device)
                final_embedding[family].append(embedding)
                c2 += 1
            c1 += 1
            print(f"{c1}/{family}:\t samples number {c2}")
    else:
        print(f"[get_malware_embeddings()]: -> Load oversized embeddings")
        fpath = os.path.join(PATH_data, "oversized_embeddings")
        fnames = [os.path.join(fpath, f) for f in os.listdir(fpath)]
        oversized = read_tensor_from_file(fnames)
        final_embedding.update(oversized)


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
    normal_embeddings = get_embeddings(data, model, tokenizer, cls_embedding=cls_embedding, pooling=mean_pooling, device=device)
    print(f"[get_malware_embeddings()]: -> embeddings retrievals done")
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
