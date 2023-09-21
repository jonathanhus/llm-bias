# Intrinsic and Extrinsic Bias Evaluation

## Get Data
`wget https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json`

`wget https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv`

## Environment Configuration
`pip install torch`

`pip install tqdm`

`pip install transformers`

TODO: create requirements.txt

## Execution

If using the original evalutate code:

`python evaluate.py --data [cp, ss] --output /Your/output/path --model [bert, roberta, albert] --method [aula, aul, cps, sss]`

If using the modified version, which doesn't require preprocessing

`TODO`