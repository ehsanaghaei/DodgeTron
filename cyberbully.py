from datasets import load_dataset
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


# --- read dataset
# train_dataset = load_dataset("poleval2019_cyberbullying", 'task01', split="train")
# test_dataset = load_dataset("poleval2019_cyberbullying", 'task01', split="test")

# data downloaded from Kaggle https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset?resource=download&select=twitter_parsed_dataset.csv
fname = "/media/ea/SSD2/Projects/DodgeTron/data/twitter_parsed_dataset.csv"
df = pd.read_csv(fname)[["Text", "oh_label"]]
df = df.dropna(subset=['Text'])
df = df.dropna(subset=['oh_label'])


def create_data_dict(row):
    return {'text': row['Text'], 'label': row['oh_label']}


data_dict = df.apply(create_data_dict, axis=1).tolist()
df = pd.DataFrame(data_dict)
dataset = Dataset.from_pandas(df)

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=25)

for t in test_dataset['text']:
    if type(t) != str:
        print(t)


# ====================== RoBERTa
# --- load tokenizer and model
def Tokenizer(data):
    return tokenizer(data['text'], padding='max_length', truncation=True)


tokenizer = RobertaTokenizer.from_pretrained(Config.model_name[0])
model = RobertaForSequenceClassification.from_pretrained(Config.model_name[0], num_labels=2)

# --- process the data
train_dataset = Dataset.from_dict(train_dataset)
test_dataset = Dataset.from_dict(test_dataset)

train_dataset = train_dataset.map(Tokenizer, batched=True)
test_dataset = test_dataset.map(Tokenizer, batched=True)


# --- label encoding
def encode_labels(data):
    data['label'] = int(data['label'])
    return data


train_dataset = train_dataset.map(encode_labels)
test_dataset = test_dataset.map(encode_labels)

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# --- config trainning
# optimizer = AdamW(model.parameters(), lr=Config.learning_rate, eps=1e-8)
# total_steps = len(train_dataloader) * Config.epochs
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Config.num_warmup_steps, num_training_steps=total_steps)

# --- config Training Args for RoBERTa
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

# ============================================== SecureBERT
if 1== 2:

    tokenizer = RobertaTokenizer.from_pretrained(Config.model_name[1])
    model = RobertaForSequenceClassification.from_pretrained(Config.model_name[1])

    # --- process the data
    train_dataset, test_dataset = train_test_split(train, test_size=0.2)

    train_dataset = Dataset.from_dict(train_dataset)
    test_dataset = Dataset.from_dict(test_dataset)
    # test_dataset = Dataset.from_dict(test_dataset)


    train_dataset = train_dataset.map(Tokenizer, batched=True)
    test_dataset = test_dataset.map(Tokenizer, batched=True)



    # --- label encoding
    def encode_labels(data):
        data['label'] = int(data['label'])
        return data


    train_dataset = train_dataset.map(encode_labels)
    test_dataset = test_dataset.map(encode_labels)


    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # --- define dataloader
    # train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    # val_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # --- config Training Args for RoBERTa
    training_args = TrainingArguments(
        output_dir=f"'./{Config.model_name[1]}-results'",
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
        etest_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # --- config Training Args for SecureBERT
    training_args = TrainingArguments(
        output_dir=f"'./{Config.model_name[1]}-results'",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        etest_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.evaluate()
    trainer.plot_metrics()
