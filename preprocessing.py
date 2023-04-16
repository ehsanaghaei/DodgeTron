import torch

def return_embedding(representation, model, tokenizer, max_seq_length=512, segment_length=400, overlap_length=100):
    data = []
    labels = []
    segments = []
    start = 0
    for family in representation:
        for id_ in representation[family]:
            text = " ".join(representation[family][id_])
            while start < len(text):
                end = min(start + segment_length, len(text))
                segment = text[start:end]
                segments.append(segment)
                if end == len(text):
                    break
                start = end - overlap_length
            # Convert each segment to input features and pass through RoBERTa model
            embeddings = []
            for segment in segments:
                tokens = tokenizer.encode(segment, add_special_tokens=True)
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                input_ids = torch.tensor(tokens).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
                embeddings.append(outputs.last_hidden_state)
            embeddings = torch.cat(embeddings, dim=1)
            data.append(embeddings)
            labels.append(family)
    return data, labels


def get_embeddings(texts, model, tokenizer, cls_embedding=True, pooling=True):
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
