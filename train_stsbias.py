from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
import argparse

# train model on STS-B dataset


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, choices=['train', 'eval'],
                    help='train or eval model')
parser.add_argument('--model', type=str,
                    help='huggingface LLM or saved checkpoint to use')
parser.add_argument('--model_dir', type=str,
                    help='when training, location to store checkpoints')



def train_model(model, model_dir):
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
    sts_bias_dataset = load_dataset("json", data_files="stsbias.json")
    print(sts_bias_dataset)


if __name__ == "__main__":
    args = parser.parse_args()
    model_checkpoint = args.model
    print(model_checkpoint)
    if args.task == 'train':
        train_model(args.model, args.model_dir)
    elif args.task == 'eval':
        preds, true_labels, gender = eval_model()
        