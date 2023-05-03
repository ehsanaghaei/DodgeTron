from prefixspan import PrefixSpan
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch
import time
from gsp import GSP
from elephant.spade import spade



def load_model_NN(model_name, base_model=True, device='cpu'):
    print(f"[load_model_NN()] -> device is being loaded on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if base_model:
        model = AutoModel.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    return model, tokenizer


def clustering(embeddings, clustering_type="kmeans", n_clusters=5, seed=13):
    from sklearn.cluster import KMeans
    if clustering_type == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=seed).fit(embeddings)

    return clustering_model


# def hir_clustering(embeddings_arrays):
#     from scipy.cluster.hierarchy import linkage
#     from sklearn.metrics import homogeneity_score
#     # assume you want to cluster the embeddings for the key "key1"
#     key = "key1"
#     embeddings_array = embeddings_arrays[key]
#
#     Z = linkage(embeddings_array, method="ward", metric="euclidean")
#
#     # assume the labels for each embedding are stored in a list called `labels`
#     homogeneity = homogeneity_score(labels, clusters)

def call_indexer(representations):
    calls = []
    for family in representations:
        for id_ in representations[family]:
            seq = representations[family][id_]
            calls += seq
            calls = list(set(calls))
    call2idx = {call: i for i, call in enumerate(calls)}
    udx2call = {i: call for i, call in enumerate(calls)}
    return call2idx, udx2call


def encode_sequence(call2idx, sequence):  # -> returns the patterns for a list of calls
    return [call2idx[call] for call in sequence]


def batch_encode_sequence(call2idx, sequences):
    return [encode_sequence(call2idx, sequence) for sequence in sequences]




def pattern_extractor(encoded_sequences, method):
    min_sup = 0.01
    max_pat = 10
    min_len = 2
    max_len = 5
    if method == "PrefixSpan":
        start_time = time.time()
        ps = PrefixSpan(encoded_sequences)
        patterns_ps = ps.frequent(min_sup)
        end_time = time.time()
        print("PrefixSpan found", len(patterns_ps), "patterns in", round(end_time - start_time, 2), "seconds")

        return patterns_ps
    elif method == "GSP":
        start_time = time.time()
        gsp = GSP(encoded_sequences)
        patterns_gsp = gsp.run(min_sup=min_sup, max_pattern=max_pat, min_len=min_len, max_len=max_len)
        end_time = time.time()
        print("GSP found", len(patterns_gsp), "patterns in", round(end_time - start_time, 2), "seconds")

        return patterns_gsp
    elif method == "Spade":
        start_time = time.time()
        spade1 = spade(encoded_sequences)
        patterns_spade = spade1.run_sup(max_sup=min_sup, max_pattern=max_pat, min_len=min_len, max_len=max_len)
        end_time = time.time()
        print("SPADE found", len(patterns_spade), "patterns in", round(end_time - start_time, 2), "seconds")

        return patterns_spade

    else:
        print('The method has not been recognized. Use "GSP", "PrefixSpan", or "Spade"')


def synthesizer(representations, method):
    call2idx, udx2call = call_indexer(representations)
    patterns = {}
    for family in representations:
        seqs = [representations[family][id_] for id_ in representations[family]]
        seqs_encoded = [encode_sequence(call2idx, sequence) for sequence in seqs]
        patterns[family] = pattern_extractor(seqs_encoded, method)
    return patterns

p = synthesizer(representations, 'PrefixSpan')
