import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['train', 'eval'],
                    help='train or eval model')
parser.add_argument('--model', type=str,
                    help='huggingface LLM or saved checkpoint to use')
parser.add_argument('--dir', type=str,
                    help='directory to save training checkpoints')

args = parser.parse_args()

# Specify model from huggingface or local checkpoint to use
# model_checkpoint = 'bert-base-uncased'
# model_checkpoint = 'checkpoint-12000'
model_checkpoint = args.model


# Read pickle that was created using script from biosbias repo
# and create Dataset
x = pd.read_pickle(r'BIOS.pkl')
data_files = Dataset.from_pandas(pd.DataFrame(data=x))

# Get all job titles and counts of each in the training set
title_count = defaultdict(int)
for record in data_files:
    title_count[record["title"]] += 1

# Create label mappings
# id: integer representing occupation (title)
# label: text representing occupation (title)
id2label = dict()
for idx, title in enumerate(sorted(title_count.keys())):
    id2label[idx] = title

label2id = dict()
for idx, title in id2label.items():
    label2id[title] = idx


# Split dataset into train, dev, and test
biosbias_dataset = data_files.train_test_split(train_size=0.8, seed=42)
dev_test_split = biosbias_dataset['test'].train_test_split(train_size=0.5, seed=42)
biosbias_dataset['dev'] = dev_test_split['train']
biosbias_dataset['test'] = dev_test_split['test']

# Add label ID (i.e., the integer) to the training set
def add_label_id(example):
    return {"label": label2id[example['title']]}

biosbias_dataset = biosbias_dataset.map(add_label_id)

# Create reduced bio
def truncate_bio(example):
    start_pos = example["start_pos"]
    return {"reduced_bio": example["raw"][start_pos:]}

biosbias_dataset = biosbias_dataset.map(truncate_bio)

# Show 3 random samples from dataset
# data_sample = biosbias_dataset["train"].shuffle(seed=42).select(range(1000))
# print(data_sample[:3])

# Define model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=28, id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Tokenize the sample text
def tokenize_function(example):
    return tokenizer(example["reduced_bio"])

tokenized_datasets = biosbias_dataset.map(tokenize_function, batched=True)

# Define a data collator to do batch padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def train_model():
    # TODO: add argument for this parameter
    # training_args = TrainingArguments('/scratch/jhus/test-trainer')
    training_args = TrainingArguments(args.dir)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


def eval_model():
    preds = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    model.to(device)
    model.eval()

    clean_tokenized_dataset = tokenized_datasets['test']
    clean_tokenized_dataset = clean_tokenized_dataset.remove_columns(["path", "raw", "name", "raw_title", "gender", 
                                                                      "start_pos", "title", "URI", "bio", "reduced_bio"])
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
    
    # Calculate bias by occupation
    for title in range(28):
        for gender in ['M', 'F']:
            total_num = len(df[(df['Labels']==title) & (df['Gender']==gender)])
            num_correct = len(df[(df['Labels']==title) & (df['Gender']==gender) & (df['Predictions']==df['Labels'])])
            print(f"Title: {id2label[title]}   Gender: {gender}   Total: {total_num}   Correct: {num_correct}   Percent: {num_correct/total_num}")

    # Calculate bias across all occupations
    print("Scores across all occupations")
    for gender in ['M', 'F']:
        total_num = len(df[df['Gender']==gender])
        num_correct = len(df[(df['Gender']==gender) & (df['Predictions']==df['Labels'])])
        print(f"Gender: {gender} Total: {total_num}  Correct: {num_correct}  Percent: {num_correct/total_num}")
    
if __name__ == "__main__":
    args = parser.parse_args()
    model_checkpoint = args.model
    print(model_checkpoint)
    if args.task == 'train':
        train_model()
    elif args.task == 'eval':
        preds, true_labels, gender = eval_model()
        calc_extrinsic_bias(preds, true_labels, gender)
