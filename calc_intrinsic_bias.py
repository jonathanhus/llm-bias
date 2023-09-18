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

        # Analyze whether BLANK is always replaced by a single word
        # for example in output:
        #     context_word_count = len(example['context'].split())
        #     stereo_word_count = len(example['stereotype'].split())
        #     anti_word_count = len(example['anti-stereotype'].split())
        # if context_word_count != stereo_word_count and context_word_count != anti_word_count:
        #     print(example)

        return output


def compute_stereoset_score():
    pass

def get_word_token_differences(tokens1, tokens2):
    list1 = list(tokens1)
    list2 = list(tokens2)

    differing_spans = []

    start = None

    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            if start is None:
                start = i
        elif start is not None:
            differing_spans.append((start, i))
            start = None

    if start is not None:
        differing_spans.append((start, max(len(list1), len(list2))))

    return differing_spans

def compute_bias(metric):
    # Set up desired model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Preprocess input data
    input_data = get_input_data()
    print(len(input_data))
    # print(input_data[10])
    # for example in input_data:
    #     print(example['context'])
    stereo_sentence = input_data[100]['stereotype']
    anti_sentence = input_data[100]['anti-stereotype']
    
    print(stereo_sentence)
    print(anti_sentence)

    stereo_encoding = tokenizer(stereo_sentence)
    anti_encoding = tokenizer(anti_sentence)
    print(stereo_encoding['input_ids'])
    print(stereo_encoding.tokens())
    print(anti_encoding['input_ids'])
    print(anti_encoding.tokens())
    print(tokenizer.mask_token_id)


if __name__ == "__main__":
    args = parser.parse_args()
    # compute_bias(args.metric)
    # if args.metric == 'sss':
    #     compute_stereoset_score()

    a = "The chess player was hispanic."
    b = "The chess player was asian."
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [1, 2, 3, 144, 555, 5, 6, 777, 8]

    span = get_word_token_differences(a, b)
    print(span)
    print(a[span[0]:span[1]])
    