import pickle
import pandas as pd
import argparse
import torch
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, AutoConfig
from transformers import TrainingArguments, Trainer
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['train', 'eval'],
                    help='train or eval model')
parser.add_argument('--model', type=str,
                    help='huggingface LLM or saved checkpoint to use')


# model_checkpoint = 'bert-base-uncased'
model_checkpoint = 'checkpoint-12000'



# Read pickle that was created using script from biosbias repo
# and create Dataset
x = pd.read_pickle(r'BIOS.pkl')
# df = pd.DataFrame(x)
data_files = Dataset.from_pandas(pd.DataFrame(data=x))

# Get all job titles and counts of each in the training set
title_count = defaultdict(int)
for record in data_files:
    title_count[record["title"]] += 1
# print(title_count)
# print(f"Num of titles: {len(title_count.keys())}")

# Create label mappings
id2label = dict()
for idx, title in enumerate(sorted(title_count.keys())):
    id2label[idx] = title
# id2label = sorted(title_count.keys())
# print(id2label)

label2id = dict()
for idx, title in id2label.items():
    label2id[title] = idx
# print(label2id)



# Split dataset into train, dev, and test
biosbias_dataset = data_files.train_test_split(train_size=0.8, seed=42)
dev_test_split = biosbias_dataset['test'].train_test_split(train_size=0.5, seed=42)
biosbias_dataset['dev'] = dev_test_split['train']
biosbias_dataset['test'] = dev_test_split['test']

# biosbias_dataset = biosbias_dataset.rename_column(original_column_name='title',
#                                                   new_column_name='label')

# Function to add label ID to the training set
def add_label_id(example):
    return {"label": label2id[example['title']]}

biosbias_dataset = biosbias_dataset.map(add_label_id)

# Show 3 random samples from dataset
# data_sample = biosbias_dataset["train"].shuffle(seed=42).select(range(1000))
# print(data_sample[:3])

# Define model and tokenizer
config = AutoConfig.from_pretrained(model_checkpoint, num_labels=28, label2id=label2id, id2label=id2label)
# model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=28, id2label=id2label, label2id=label2id)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print("Here")

# Tokenize the sample text
def tokenize_function(example):
    start_pos = example["start_pos"]
    return tokenizer(example["raw"][start_pos:])

tokenized_datasets = biosbias_dataset.map(tokenize_function, batched=False)
# print(tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def train_model():
    # TODO: add argument for this parameter
    training_args = TrainingArguments('/scratch/jhus/test-trainer')

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Once training is complete, do eval


def eval_model():
    preds = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    clean_tokenized_dataset = tokenized_datasets['test']
    clean_tokenized_dataset = clean_tokenized_dataset.remove_columns(["path", "raw", "name", "raw_title", "gender", 
                                                                      "start_pos", "title", "URI", "bio"])
    eval_dataloader = DataLoader(clean_tokenized_dataset, batch_size=8, collate_fn=data_collator)
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        preds.extend(predictions.tolist())
    return preds, tokenized_datasets['test']['label'], tokenized_datasets['test']['gender']
        # metric.add_batch(predictions=predictions, references=batch["labels"])

def calc_extrinsic_bias(prediction, true_label, gender):
    df = pd.DataFrame(list(zip(prediction, true_label, gender)),
                      columns=['Predictions', 'Labels', 'Gender'])
    for title in range(28):
        for gender in ['M', 'F']:
            j = len(df[(df['Labels']==title) & (df['Gender']==gender)])
            print(f"Title: {title}   Gender: {gender}   Num: {j}")
    
if __name__ == "__main__":
    args = parser.parse_args()
    if args.task == 'train':
        train_model()
    elif args.task == 'eval':
        preds, true_labels, gender = eval_model()
        # print(preds)
        # print(true_labels)
        # print(gender)
        calc_extrinsic_bias(preds, true_labels, gender)
