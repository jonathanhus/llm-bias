# 

import json
import argparse
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, choices=['sss'],
                    help='score to evaluate')
parser.add_argument('--model', type=str,
                    help='huggingface LLM to use')

def get_input_data():
    with open('dev.json') as f:
        all_data = json.load(f)
        output = []
        for ss_sample in all_data['data']['intrasentence']:
            output_sample = dict()
            output_sample['bias_type'] = ss_sample['bias_type']
            output_sample['context'] = ss_sample['context']
            for sentence in ss_sample['sentences']:
                label = sentence['gold_label']
                output_sample[label] = sentence['sentence']
            output.append(output_sample)

        # print(output[999])
        return output


def compute_stereoset_score():
    pass



def compute_bias(metric):
    # Set up desired model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Preprocess input data
    input_data = get_input_data()
    print(len(input_data))
    for example in input_data:
        print(example['context'])


if __name__ == "__main__":
    args = parser.parse_args()
    compute_bias(args.metric)
    # if args.metric == 'sss':
    #     compute_stereoset_score()
    