# Cross-lingual offensive language identification

## Installation Python 3.6

To be able to run this project you need to run following command:
```bash
 pip install -r requirements.txt
```
To run ELMo models you need CUDA 10.0.
To run mBERT and XLM-RoBERTa you need Torch 1.8.1 and CUDA 10.2, please https://pytorch.org/ for installation instructions.

Models are available here: https://drive.google.com/file/d/1-epv8kbiSAH9VFHRmyflOVxyCCqoYKI5/view?usp=sharing

Contact us for datasets (ec9381@student.uni-lj.si) or use the transform_dataset.py script to transform the following datasets - whitesupremacy.csv, twitter.csv, reddit.csv, gab.csv, fox-news.json, CONAN.json, which are the datasets used as described in the report and use the Twitter API to obtain tweets from the Slovenian Twitter hate speech dataset IMSyPP-sl

# Instructions
In order to reproduce the results written in the Paper run following command(s)
## Bert Classification
```bash
 python classification.py --model bert --language {slovene|english} --type {binary|multilabel}
```
## Elmo Classification
```bash
 python classification.py --model elmo --language {slovene|english} --type {binary|multilabel}
```

## TFIDF Classification
```bash
 python classification.py --model tfidf --language {slovene|english} --type {binary|multilabel}
```

## CUSTOM CLASSIFIER Classification
```bash
 python classification.py --model custom-classifier --language {slovene|english} --type {binary|multilabel}
```

## XLM Classification
```bash
 python classification.py --model xlm --language {slovene|english} --type {binary|multilabel}
```
