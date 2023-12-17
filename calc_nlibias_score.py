'''
    Calcuates NLI-Bias score based on procedure described in 
    Debiasing ins't enough! paper
    
    Evaluation data is in nli_bias.csv
    There are 6 gendered words (3 male and 3 female):
        gentleman, guy, man
        lady, girl, woman

    0: entailment, 1: neutral, 2: contradiction
'''
import pprint
from collections import defaultdict
with open('/Users/jonhus/Downloads/nlibias_results_bert_debiased.csv') as f:
    results = f.readlines()

nums = defaultdict(lambda: defaultdict(int))

for line in results:
    category, gender = line.strip().split(',')
    nums[gender][category] += 1

print(nums)
pprint.pprint(nums)
print("Done")