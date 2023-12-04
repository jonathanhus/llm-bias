# Intrinsic and Extrinsic Bias Evaluation

This repo contains code to evaluate the intrinsic and extrinsic bias in NLP large language models.
## Environment Configuration

`pip install -r requirements.txt`


## Intrinsic Bias

The calculation of intrinsic bias is based largely on the work of Masahiro Kaneko. The setup and use of scripts is described below. More information can be found at https://github.com/kanekomasahiro/evaluate_bias_in_mlm


### Get Data

The following commands download and preprocess the CrowS=Pairs and Stereoset data. If desired, these steps can be skipped since the preprocessed output is included in the repo.
```
mkdir data
wget -O data/cp.csv https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv
wget -O data/ss.json https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json
python -u preprocess.py --input crows_pairs --output data/paralled_cp.json
python -u preprocess.py --input stereoset --output data/paralled_ss.json
```

### Evaluation
To perform evaluation on LLMs, run the following command:

`python evaluate.orig.py --data [cp, ss] --output /Your/output/path --model [bert, roberta, albert] --method [aula, aul, cps, sss]`


## Extrinsic Bias

Extrinsic bias is calculated using three different datasets and metrics: BiosBias, STS-Bias, and NLI-Bias.

### BiosBias
The biosbias dataset was generated following the instructions located here:
https://github.com/microsoft/biosbias

For convenience, the BIOS.pkl that was generated is included in this repo

To finetune the model on the BiosBias dataset:

`python train_biosbias.py --task train --model [bert-base-uncased, albert-base-v2, roberta-base] --model_dir /model/save/path`

To evaluate the finetuned model on the BiosBias test dataset, specifying a model checkpoint from the previous finetuning step:

`python train_biosbias.py --task eval --model /scratch/jhus/test-trainer/checkpoint-12000/`

### STS-Bias
The STS-Bias dataset is not publicly available. As the authors did, we attempted to recreate the dataset following the procedure briefly described in 
"Measuring and Reducing Gendered Correlations in Pre-trained Models" by Kellie Webster, et al. https://arxiv.org/pdf/2010.06032.pdf

BLS occupation statistics can be found here: https://github.com/rudinger/winogender-schemas/blob/master/data/occupations-stats.tsv

We downloaded the STS-B data from here: http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz

To generate the STS-bias dataset:

`python create_stsbias_dataset.py`

The generated dataset is called "stsbias.json" and is included in this repo for convenience.

To finetune the model on the STS-Bias dataset

`python train_stsbias.py --task train --model bert-base-uncased --model_dir /your/output/dir`

To evaluate the finetuned model on the STS-Bias test dataset, specifying a model checkpoint from the previous finetuning step:

python train_stsbias.py --task eval --model /scratch/jhus/test-trainer-sts/checkpoint-2000/

### NLI-Bias
The NLI-Bias dataset was generated using scripts and files located here:
https://github.com/sunipa/On-Measuring-and-Mitigating-Biased-Inferences-of-Word-Embeddings.git

`python generate_templates.py --noun --p occupations --h gendered_words --output nli_bias.csv`

The generated output file is over 400MB in size. To reduce the size by removing unnecessary columns and limiting the number of rows, run the following script:

`python create_nli_dataset.py`

For convenience the generated dataset is included in this repo as `reduced_nli_bias.csv`

To finetune the model on the NLI-Bias dataset

`python train_nlibias.py --task train --model bert-base-uncased --model_dir /scratch/jhus/test-trainer-nli`

To evaluate the finetuned model on the STS-Bias test dataset, specifying a model checkpoint from the previous finetuning step:

`python train_nlibias.py --task eval --model /saved/model/checkpoint/`


## Debiased Models
Debiased models are obtained using code from the following repo:

https://github.com/Irenehere/Auto-Debias.git

The debiased models can then be finetuned and evaluated on the BiosBias, STS_Bias, and NLI-Bias datasets using the commands and scripts mentioned above, substituting the appropriate model in the command parameters.

## SLURM Scripts
A number of sample SLURM scripts are included in this repo for instances when the models are to be trained and evaluated on cluster environments.