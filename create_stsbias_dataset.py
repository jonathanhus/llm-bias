import pandas as pd
import re
import json

'''
Create the STS-Bias dataset
'''

with open('stsbenchmark/sts-test.csv') as f:
    lines = f.readlines()

# Get all sentences that start with "A man" or "A woman"
sst2_sentences = []
for idx, line in enumerate(lines):
    sentence = line.split('\t')[5].strip()
    if sentence.startswith("A man") or sentence.startswith("A woman"):
        sst2_sentences.append(sentence)
    sentence = line.split('\t')[6].strip()
    if sentence.startswith("A man") or sentence.startswith("A woman"):
        sst2_sentences.append(sentence)

# Remove any duplicate sentences
sst2_sentences = set(sst2_sentences)

templates = []
for sentence in sst2_sentences:
    if sentence.startswith("A man and a woman"):
        continue
    elif sentence.startswith("A man and woman"):
        continue
    elif sentence.startswith("A woman and man"):
        continue
    elif sentence.startswith("A woman and a man"):
        continue
    elif "beard" in sentence:
        continue
    elif "his" in sentence:
        continue
    elif "himself" in sentence:
        continue
    elif "her" in sentence:
        continue
    elif "herself" in sentence:
        continue
    elif "man, woman" in sentence:
        continue
    elif "top hat" in sentence:
        continue
    else:
        templates.append(sentence)

clean_templates = []
for sentence in templates:
    if sentence.startswith("A man"):
        if "woman" in sentence:
            continue
        else:
            clean_templates.append(sentence)
    if sentence.startswith("A woman"):
        # print(sentence)
        clean_templates.append(sentence)


with open('occupations.txt') as f:
    lines = f.readlines()

occupations = []
for line in lines:
    occupations.append(line.strip())

examples = []
for template in clean_templates:
    for occupation in occupations:
        if template.startswith("A man"):
            male_sentence = template
            female_sentence = template.replace("A man", "A woman")
            occupation_sentence = template.replace("A man", f"A {occupation}")
        elif template.startswith("A woman"):
            male_sentence = template.replace("A woman", "A man")
            female_sentence = template
            occupation_sentence = template.replace("A woman", f"A {occupation}")
        else:
            print("ERROR")
        example = dict()
        example["male_sentence"] = male_sentence
        example["female_sentence"] = female_sentence
        example["occupation_sentence"] = occupation_sentence
        example["occupation"] = occupation
        examples.append(example)


# Write dataset to file in jsonl format
with open('stsbias.json', 'w') as outfile:
    for example in examples:
        json.dump(example, outfile)
        outfile.write('\n')
