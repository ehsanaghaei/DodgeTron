from initialize_models import load_model_NN
from paths import PATH_data, PATH_infostealer
from preprocessing import  get_embeddings, get_malware_embeddings
from read_data import list_files, read_files, build_representation, rep_summarizer, build_representation_batch

if __name__ == "__main__":
# ------------------	 Configs 	------------------
    class Config:
        model_name = "ehsanaghaei/SecureBERT"
        mode= "batch"

    # ------------------	 Read Data 	------------------
    step_ = "Read File"
    print(f"[###### {step_}] -> Read malware json files")
    families, fnames = list_files(PATH_infostealer)
    # malwares = read_files(fnames)
    print(f"[###### {step_}] -> Convert to sequence representation")
    # representations = build_representation(malwares, segments=['api'])
    representations= build_representation_batch(fnames[:10], segments=['api'], joined=False, save=True)

    print(f"[###### {step_}] -> Summarize the representation")
    representations_sum = rep_summarizer(representations)

    # ------------------	 Data Preparation 	------------------
    step_ = "Data Preparation"
    print(f"[###### {step_}] -> Load model and tokenizer")
    model, tokenizer = load_model_NN(Config.model_name, base_model=True)
    print(f"[###### {step_}] -> Get malware embeddings")
    rep_norm, rep_oversize = get_malware_embeddings(representations, model, tokenizer, cls_embedding=True, mean_pooling=True, max_size=350, overlap_size=100)
    # data, labels = return_embedding(representations_sum, model, tokenizer, max_seq_length=512, segment_length=400, overlap_length=100)
    # text = " ".join(representations_sum['emotet']['d8ea223ed89c83d10ef935a77c3a2e5ecb5e5817dea7e9f459eaee17a9a90202.emotet'])
    # text2 = " ".join(representations_sum['emotet']['aeca6bcf2db969cb99ee3819cba054507effe800e24c8db41bb5a29a24a89102.emotet'])
    # a = get_embeddings(text, model, tokenizer)
    # b = get_embeddings('my name is Ehsan', model, tokenizer, cls_embedding=True, pooling=True)