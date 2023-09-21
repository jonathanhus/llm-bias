# 

import json
import argparse
import difflib
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, choices=['sss'],
                    help='score to evaluate')
parser.add_argument('--model', type=str,
                    help='huggingface LLM to use')

def get_input_data():
    # Input data for StereoSet
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

def get_rank_for_gold_token(log_probs, token_ids):
    '''
    Get rank for gold token from log probability.
    '''
    sorted_indexes = torch.sort(log_probs, dim=1, descending=True)[1]
    ranks = torch.where(sorted_indexes == token_ids)[1] + 1
    ranks = ranks.tolist()

    return ranks

def compute_stereoset_score(model, token_ids, spans, mask_id):
    masked_token_ids = token_ids.clone()
    masked_token_ids[:, spans] = mask_id
    hidden_states = model(masked_token_ids)
    hidden_states = hidden_states['logits'].squeeze(dim=0)
    token_ids = token_ids.view(-1)[spans]
    log_softmax = torch.nn.LogSoftmax(dim=1)
    log_probs = log_softmax(hidden_states)[spans]
    span_log_probs = log_probs[:,token_ids]
    score = torch.mean(span_log_probs).item()

    if log_probs.size(0) != 0:
        ranks = get_rank_for_gold_token(log_probs, token_ids)
    else:
        ranks = [-1]
    
    return score, ranks

# def get_word_token_differences(tokens1, tokens2):
#     list1 = list(tokens1)
#     list2 = list(tokens2)

#     differing_spans = []

#     start = None

#     for i in range(min(len(list1), len(list2))):
#         if list1[i] != list2[i]:
#             if start is None:
#                 start = i
#         elif start is not None:
#             differing_spans.append((start, i))
#             start = None

#     if start is not None:
#         differing_spans.append((start, max(len(list1), len(list2))))

#     return differing_spans

def get_word_token_differences(tokens1, tokens2, operation):
    '''
        Calculates tokens that are common and different between
        two sequences of tokens.
        Returns the indices of each sequence that are common or different
        Args:
            tokens1, tokens2: list of tokens_ids
            operation: "equal" or "diff", identifies whether to find
                        equal (unmodified) or unequal (modified) tokens
                        in the two lists
    '''
    seq1 = [str(x) for x in tokens1.tolist()]
    seq2 = [str(x) for x in tokens2.tolist()]
    # seq1 = [str(x) for x in tokens1]
    # seq2 = [str(x) for x in tokens2]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if (operation == 'equal' and op[0] == 'equal') \
                or (operation == 'diff' and op[0] != 'equal'):
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2

def compute_bias(metric):
    # Set up desired model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Preprocess input data
    input_data = get_input_data()
    # print(len(input_data))
    # print(input_data[10])
    for example in input_data:
        # print(example['context'])
        stereo_sentence = example['stereotype']
        anti_sentence = example['anti-stereotype']
        
        # print(stereo_sentence)
        # print(anti_sentence)

        stereo_encoding = tokenizer(stereo_sentence, return_tensors='pt')
        anti_encoding = tokenizer(anti_sentence, return_tensors='pt')
        # print(stereo_encoding['input_ids'])
        # print(stereo_encoding.tokens())
        # print(anti_encoding['input_ids'])
        # print(anti_encoding.tokens())

        # Get list of modified and unmodified tokens
        stereo_modified_tokens, anti_modified_tokens = get_word_token_differences(stereo_encoding['input_ids'][0],
                                                                                   anti_encoding['input_ids'][0],
                                                                                   'diff')
        # print(stereo_modified_tokens, anti_modified_tokens)
        stereo_score, anti_ranks = compute_stereoset_score(model, 
                                                      stereo_encoding['input_ids'],
                                                      stereo_modified_tokens,
                                                      tokenizer.mask_token_id)
        anti_score, stereo_ranks = compute_stereoset_score(model,
                                                           anti_encoding['input_ids'],
                                                           anti_modified_tokens,
                                                           tokenizer.mask_token_id)
    all        


if __name__ == "__main__":
    args = parser.parse_args()
    compute_bias(args.metric)
    # if args.metric == 'sss':
    #     compute_stereoset_score()

    # a = "The chess player was hispanic."
    # b = "The chess player was asian."
    # a = [1, 2, 3, 4, 5, 6, 7, 8]
    # b = [1, 2, 3, 144, 555, 5, 6, 777, 8]
    # a = torch.Tensor(a)
    # b = torch.Tensor(b)

    # span = get_word_token_differences(a, b, "equal")
    # print(span)
    # print(a[span[0]:span[1]])
    