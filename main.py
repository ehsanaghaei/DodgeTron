import torch

from initialize_models import load_model_NN
from paths import PATH_data, PATH_infostealer
from preprocessing import get_embeddings, get_malware_embeddings, handle_large_input_embedding
from read_data import list_files, read_files, build_representation, rep_summarizer, build_representation_batch

if __name__ == "__main__":
    # ------------------	 Configs 	------------------
    class Config:
        model_name = "ehsanaghaei/SecureBERT"
        mode = "batch"
        load_representations = True


    # ------------------	 Read Data 	------------------
    step_ = "Read File"
    print(f"[###### {step_}] -> Read malware json files")
    families, fnames = list_files(PATH_infostealer)
    # malwares = read_files(fnames)
    print(f"[###### {step_}] -> Convert to sequence representation")
    # representations = build_representation(malwares, segments=['api'])
    representations = build_representation_batch(fnames, segments=['api'], joined=False, save=True, load=Config.load_representations)
    print(f"[###### {step_}] -> Summarize the representation")
    representations = rep_summarizer(representations)

    # ------------------	 Data Preparation 	------------------
    step_ = "Data Preparation"
    print(f"[###### {step_}] -> Load model and tokenizer")
    model, tokenizer = load_model_NN(Config.model_name, base_model=True)
    print(f"[###### {step_}] -> Get malware embeddings")

    rep_norm, rep_oversize = get_malware_embeddings(representations, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=350, overlap_size=100)
    # text = 300*"this is test text"
    # input_ids = tokenizer.encode(text, padding=True, truncation=False, return_tensors='pt')[0]
    # l = handle_large_input_embedding(model, input_ids, window_size=300, overlap_size=128, cls_embedding=True, pooling=True)