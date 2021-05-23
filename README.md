# Cross-lingual offensive language identification

## Installation Python 3.6

To be able to run this project you need to run following command:
```bash
 pip install -r requirements.txt
```
To run ELMo models you need CUDA 10.0 and cuDNN 7.6.5 for CUDA 10.0.
To run mBERT and XLM-RoBERTa you need Torch 1.8.1 and CUDA 10.2, please visit https://pytorch.org/ for installation instructions.

Models are available here: https://drive.google.com/file/d/1-epv8kbiSAH9VFHRmyflOVxyCCqoYKI5/view?usp=sharing
Extract them to ./models

Our Slovene hate speech dataset is uploaded and included in the dataset folder.

Contact us for the rest of the datasets (ec9381@student.uni-lj.si) and extract them to ./ or use the transform_dataset.py script to transform the following datasets - whitesupremacy.csv, reddit.csv, gab.csv, fox-news.json, CONAN.json, which are the datasets used as described in the report and merge them together. From the article Multilingual and Multi-Aspect Hate Speech Analysis we manually extracted sexist comments with the label 'women' in the 'group' column. Use the Twitter API to obtain tweets from the Slovenian Twitter hate speech dataset IMSyPP-sl. Then use preprocess_dataset.py on the English merged dataset to split it into training and test sets and similarly on the Slovenian Twitter hate speech dataset.



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
