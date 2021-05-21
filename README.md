# Cross-lingual offensive language identification

## Installation Python 3.7

To be able to run this project you need to run following command:
```bash
 pip install -r requirements.txt
```

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
