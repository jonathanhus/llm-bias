from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse

'''
    Script to finetune a model and then evaluate it for bias using STS-Bias data   
'''


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['train', 'eval'],
                    help='train or eval model')
parser.add_argument('--model', type=str,
                    help='huggingface LLM or saved checkpoint to use')
parser.add_argument('--model_dir', type=str,
                    help='when training, location to store checkpoints')



def train_model(model, model_dir):
    '''
        Train the model to be evaluated for bias. The model is trained
        on the STS-B dataset, which is included in the GLUE Benchmark
    '''
    raw_datasets = load_dataset('glue', 'stsb')

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"])
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    training_args = TrainingArguments(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=1)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    

def eval_model(model):
    '''
        Evaluate model on STS-Bias dataset
        stsbias.json must have been created previously
        using the script create_stsbias_dataset.py)
    '''
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Load sts-bias dataset
    sts_bias_dataset = load_dataset("json", data_files="stsbias.json")
    print(sts_bias_dataset)

    def tokenize_male_function(example):
        return tokenizer(example["male_sentence"], example["occupation_sentence"])
    
    def tokenize_female_function(example):
        return tokenizer(example["female_sentence"], example["occupation_sentence"])

    # Create male and female datasets
    male_tokenized_dataset = sts_bias_dataset.map(tokenize_male_function, batched=True)
    male_tokenized_dataset = male_tokenized_dataset.remove_columns(["male_sentence", "female_sentence", "occupation_sentence", "occupation"])
    female_tokenized_dataset = sts_bias_dataset.map(tokenize_female_function, batched=True)
    female_tokenized_dataset = female_tokenized_dataset.remove_columns(["male_sentence", "female_sentence", "occupation_sentence", "occupation"])

    # Define a data collator to do batch padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # First evaluate similarity scores of male and occupation
    eval_dataloader = DataLoader(male_tokenized_dataset["train"], batch_size=8, collate_fn=data_collator)
    male_preds = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits.squeeze()
        male_preds.extend(logits.tolist())

    # Second evaluate similarity scores of female and occupation
    eval_dataloader = DataLoader(female_tokenized_dataset["train"], batch_size=8, collate_fn=data_collator)
    female_preds = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits.squeeze()
        female_preds.extend(logits.tolist())

    # Save results (which will be analyzed using Excel)
    results = pd.DataFrame(list(zip(male_preds, female_preds, sts_bias_dataset['train']['occupation'])),
                           columns=["Male Score", "Female Score", "Occupation"])
    results.to_csv('stsbias_results.csv', index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    model_checkpoint = args.model
    print(model_checkpoint)
    if args.task == 'train':
        train_model(args.model, args.model_dir)
    elif args.task == 'eval':
        eval_model(args.model)
        