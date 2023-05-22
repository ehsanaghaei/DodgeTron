from bertopic import BERTopic
from prefixspan import PrefixSpan
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch
import time
# from gsp import GSP
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

def load_BERTopic(model_name):
    from transformers.pipelines import pipeline

    embedding_model = pipeline("feature-extraction", model=model_name)
    topic_model = BERTopic(embedding_model=embedding_model)


def hir_clustering(embedding_dict,n_clusters=18):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import homogeneity_score
    import numpy as np
    all_vectors = []
    labels = []
    for key, vectors in embedding_dict.items():
        all_vectors.extend(vectors)
        labels.extend([key] * len(vectors))

    # Convert the list of vectors to a NumPy array
    all_vectors = np.array(all_vectors)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    predicted_labels = clustering.fit_predict(all_vectors)

    # Calculate the homogeneity score
    score = homogeneity_score(labels, predicted_labels)
    print("Homogeneity score:", score)


def agglo(embedding_dict):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import homogeneity_score

    # Assuming your dictionary is called 'embedding_dict'
    # and each key's value is a list of embedding vectors

    # Combine all embedding vectors into a single list
    all_vectors = []
    labels = []
    for key, vectors in embedding_dict.items():
        all_vectors.extend(vectors)
        labels.extend([key] * len(vectors))

    # Convert the list of vectors to a NumPy array
    all_vectors = np.array(all_vectors)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=9)
    predicted_labels = clustering.fit_predict(all_vectors)

    # Plot dendrogram
    linked = linkage(all_vectors, method='ward', metric='euclidean')
    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

    # Calculate the homogeneity score
    score = homogeneity_score(labels, predicted_labels)
    print("Homogeneity score:", score)
    return predicted_labels


def agglo_ncluster(embedding_dict, n_clusters_values=None):
    if n_clusters_values is None:
        n_clusters_values = [9, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import homogeneity_score
    all_vectors = []
    labels = []
    for key, vectors in embedding_dict.items():
        all_vectors.extend(vectors)
        labels.extend([key] * len(vectors))

    # Convert the list of vectors to a NumPy array
    all_vectors = np.array(all_vectors)

    # Define the values of n_clusters to test
    n_clusters_values = n_clusters_values

    # Initialize an empty list to store the homogeneity scores
    scores = []

    # Perform hierarchical clustering and calculate homogeneity score for each n_clusters value
    for n_clusters in n_clusters_values:
        if n_clusters == 0:
            n_clusters = 1
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        predicted_labels = clustering.fit_predict(all_vectors)
        score = homogeneity_score(labels, predicted_labels)
        print(f"Hemogeneity Score for {n_clusters}: {score}")
        scores.append(score)


    # Plot the homogeneity scores
    plt.figure(figsize=(8, 5))
    plt.plot(n_clusters_values, scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Score vs. Number of Clusters')
    plt.grid(True)
    plt.show()



