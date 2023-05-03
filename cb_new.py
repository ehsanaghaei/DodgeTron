from datasets import load_dataset
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score
from datasets import Dataset
import pandas as pd


# --- config class
class Config:
    model_name = ['roberta-base', "ehsanaghaei/SecureBERT"]  # 'roberta-base', "ehsanaghaei/SecureBERT"
    learning_rate = 2e-5
    epochs = 2
    batch_size = 6
    num_warmup_steps = 100


# --- read and prepare data

fname = "/media/ea/SSD2/Projects/DodgeTron/data/twitter_parsed_dataset.csv"
df = pd.read_csv(fname)[["Text", "oh_label"]]
df = df.dropna(subset=['Text'])
df = df.dropna(subset=['oh_label'])

df = df.rename(columns={'Text': 'text', "oh_label":"label"})

train_df, val_df = train_test_split(df, test_size=0.2)

tokenizer = RobertaTokenizer.from_pretrained(Config.model_name[0])
model = RobertaForSequenceClassification.from_pretrained(Config.model_name[0], num_labels=2)


def encode_data(data):
    # Tokenize the inputs
    inputs = tokenizer(data['text'], padding='max_length', truncation=True, max_length=128)
    # Create the encoded inputs
    inputs = {key: torch.tensor(val) for key, val in inputs.items()}
    # Add the labels
    inputs['labels'] = torch.tensor(data['label'])
    return inputs


# ---- encode data
train_encodings = train_df.apply(encode_data, axis=1)
val_encodings = val_df.apply(encode_data, axis=1)

# ---- convert to HF dataset
train_dataset = Dataset.from_pandas(train_encodings.to_frame())
val_dataset = Dataset.from_pandas(val_encodings)

# ---- defing configs
training_args = TrainingArguments(
    output_dir=f"'./{Config.model_name[0]}-results'",
    evaluation_strategy='epoch',
    num_train_epochs=Config.epochs,
    per_device_train_batch_size=Config.batch_size,
    per_device_eval_batch_size=Config.batch_size,
    learning_rate=Config.learning_rate,
    warmup_steps=Config.num_warmup_steps,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=500,
    load_best_model_at_end=True,
    save_strategy='epoch'
)


# --- define Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1, 'precision': precision}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
