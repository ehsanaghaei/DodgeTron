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
    print(f"main(): {step_}] -> Read malware json files")
    families, fnames = list_files(PATH_infostealer)
    # malwares = read_files(fnames)
    print(f"main(): {step_}] -> Convert to sequence representation")
    # representations = build_representation(malwares, segments=['api'])
    representations = build_representation_batch(fnames, segments=['api'], joined=False, save=True, load=Config.load_representations)
    # representations = build_representation_batch(fnames[:10]+fnames[990:1000]+fnames[2000:2010], segments=['api'], joined=False, save=False, load=False)
    print(f"main(): {step_}] -> Summarize the representation")
    representations = rep_summarizer(representations)

    # ------------------	 Data Preparation 	------------------
    step_ = "Data Preparation"
    print(f"main(): {step_}] -> Load model and tokenizer")
    model, tokenizer = load_model_NN(Config.model_name, base_model=True)
    print(f"main(): {step_}] -> Get malware embeddings")

    rep_norm, rep_oversize = get_malware_embeddings(representations, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=512, overlap_size=100)
    text = representations['emotet']['b60c0c2050d1f99ef73709f977a213a30b6e02a79c7a22515f848c1702c9edff.emotet']
    text = " ".join(text)
    input_ids2 = tokenizer.encode(text, padding=True, truncation=False, return_tensors='pt')[0]
    output_tensors = torch.split(input_ids2, 512)
    input_strings = [' '.join([str(x.item()) for x in tensor]) for tensor in output_tensors]
    inputs = tokenizer.batch_encode_plus(input_strings, return_tensors='pt', padding=True, truncation=True)


    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = torch.mean(embeddings, dim=0)
    torch.save(embeddings, "embedding.pt")
    #
    # l = handle_large_input_embedding(model, input_ids, window_size=300, overlap_size=128, cls_embedding=True, pooling=True)