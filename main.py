import torch

from initialize_models import load_model_NN, agglo, agglo_ncluster
from paths import PATH_infostealer
from preprocessing import get_malware_embeddings
from read_data import list_files, rep_summarizer, build_representation_batch, read_tensor_from_file
from synthesize_functions import find_common_apis

if __name__ == "__main__":
    # ------------------	 Configs 	------------------
    class Config:
        model_name = "ehsanaghaei/SecureBERT"
        mode = "batch"
        load_representations = True


    # ------------------	 Read Data 	------------------
    step_ = "Read File"
    print(f"[main(): {step_}] -> Read malware json files")
    families, fnames = list_files(PATH_infostealer)
    # malwares = read_files(fnames)
    print(f"[main(): {step_}] -> Convert to sequence representation")
    # representations = build_representation(malwares, segments=['api'])
    representations = build_representation_batch(fnames, segments=['api'], joined=False, save=True, load=Config.load_representations)
    common_apis = find_common_apis(representations, 100)
    # representations = build_representation_batch(fnames[:10]+fnames[990:1000]+fnames[2000:2010], segments=['api'], joined=False, save=False, load=False)
    print(f"[main(): {step_}] -> Summarize the representation")
    representations_summary = rep_summarizer(representations)
    common_apis_summarized = find_common_apis(representations_summary, 100)

    # ------------------	 Data Synthesize 	------------------


    # ------------------	 Data Preparation 	------------------
    step_ = "Data Preparation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    print(f"[main(): {step_}] -> Load model and tokenizer")
    model, tokenizer = load_model_NN(Config.model_name, base_model=True, device=device)

    print(f"[main(): {step_}] -> Get malware embeddings")

    final_embedding = get_malware_embeddings(representations_summary, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=512, overlap_size=100, load_representations=True, load_embeddings=True, device=device)
    final_embedding['loki'] += final_embedding['Loki']
    del final_embedding['Loki']
    labels = agglo_ncluster(final_embedding,list(range(0,2000,50)))

    # ------------------	 Modeling 	------------------
    from scipy.cluster.hierarchy import linkage
    from sklearn.metrics import homogeneity_score
    # assume you want to cluster the embeddings for the key "key1"
    key = "key1"
    embeddings_array = final_embedding['emotet']

    Z = linkage(embeddings_array, method="ward", metric="euclidean")
