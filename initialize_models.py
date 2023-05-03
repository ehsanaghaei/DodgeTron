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
