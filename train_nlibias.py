from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse

# train model on MNLI dataset
# eval on NLI-Bias dataset


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['train', 'eval'],
                    help='train or eval model')
parser.add_argument('--model', type=str,
                    help='huggingface LLM or saved checkpoint to use')
parser.add_argument('--model_dir', type=str,
                    help='when training, location to store checkpoints')

# Compute metric during training
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mnli")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(model, model_dir):
    raw_datasets = load_dataset('glue', 'mnli')

    tokenizer = AutoTokenizer.from_pretrained(model)
    
    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"])
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    training_args = TrainingArguments(model_dir, evaluation_strategy="epoch")
    model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=3)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation_matched"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate model performance
    predictions = trainer.predict(tokenized_datasets["validation_matched"])
    print(predictions.predictions.shape, predictions.label_ids.shape)

    preds = np.argmax(predictions.predictions, axis=-1)

    metric = evaluate.load("glue", "mnli")
    eval_metrics = metric.compute(predictions=preds, references=predictions.label_ids)
    print(eval_metrics)
    

def eval_model(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Load nli-bias dataset
    # nli_bias_dataset = load_dataset("csv", data_files="nli_bias.csv")
    nli_bias_dataset = load_dataset("csv", data_files="/scratch/jhus/nli_bias.csv")
    print(nli_bias_dataset)

    def tokenize_function(example):
        return tokenizer(example["premise"], example["hypothesis"])

    # 
    tokenized_dataset = nli_bias_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["id", "pair type", "premise_filler_word",
                                                          "hypothesis_filler_word", "template_type", "premise", "hypothesis"])
 

    # Define a data collator to do batch padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Evaluate
    eval_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, collate_fn=data_collator)
    preds = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        preds.extend(predictions.tolist())

    # Save results (which will be analyzed using Excel)
    results = pd.DataFrame(list(zip(preds, nli_bias_dataset['train']['hypothesis_filler_word'])),
                           columns=["Prediction", "GenderWord"])
    results.to_csv('nlibias_results.csv', index=False)



if __name__ == "__main__":
    args = parser.parse_args()
    model_checkpoint = args.model
    print(model_checkpoint)
    if args.task == 'train':
        train_model(args.model, args.model_dir)
    elif args.task == 'eval':
        eval_model(args.model)
        