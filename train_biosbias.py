import pickle
import pandas as pd
from torch.utils.data import random_split
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from collections import defaultdict

# Read pickle that was created using script from biosbias repo
# and create Dataset
x = pd.read_pickle(r'BIOS.pkl')
# df = pd.DataFrame(x)
data_files = Dataset.from_pandas(pd.DataFrame(data=x))

# Create label mappings 
title_count = defaultdict(int)
for record in data_files:
    title_count[record["title"]] += 1
print(title_count)
print(f"Num of titles: {len(title_count.keys())}")

id2label = sorted(title_count.keys())
print(id2label)

label2id = dict()
for idx, title in enumerate(id2label):
    label2id[title] = idx
print(label2id)



# Split dataset into train, test
biosbias_dataset = data_files.train_test_split(train_size=0.8, seed=42)
# biosbias_dataset = biosbias_dataset.rename_column(original_column_name='title',
#                                                   new_column_name='label')

def add_label_id(example):
    return {"label": label2id[example['title']]}

biosbias_dataset = biosbias_dataset.map(add_label_id)

data_sample = biosbias_dataset["train"].shuffle(seed=42).select(range(1000))
print(data_sample[:3])

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28, id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def tokenize_function(example):
    start_pos = example["start_pos"]
    return tokenizer(example["raw"][start_pos:])

tokenized_datasets = biosbias_dataset.map(tokenize_function, batched=False)
# print(tokenized_datasets)


# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
# Train model
# Goal is 