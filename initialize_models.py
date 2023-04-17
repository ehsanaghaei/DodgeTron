from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch


def load_model_NN(model_name, base_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if base_model:
        model = AutoModel.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer


def clustering(embeddings, clustering_type="kmeans", n_clusters=5, seed=13):
    from sklearn.cluster import KMeans
    if clustering_type == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=seed).fit(embeddings)

    return clustering_model

