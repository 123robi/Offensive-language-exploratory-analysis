# Cross-lingual offensive language identification

## Installation Python 3.7

To be able to run this project you need to run following command:
```bash
 pip install -r requirements.txt
```

# Custom classifiers

## Slovene dataset

Binary dataset:
```bash
 python run_custom_classifiers.py hatespeech slovene
```
Multilabel dataset 
```bash
 python run_custom_classifiers.py subtype slovene
```

---

## English dataset
Binary dataset:
```bash
 python run_custom_classifiers.py hatespeech english
```
Multilabel dataset 
```bash
 python run_custom_classifiers.py subtype english
```


# Tf-idf classifiers

## Slovene dataset

Binary dataset:
```bash
 python run_tf_idf_classifiers.py binary slovene
```
Multilabel dataset 
```bash
 python run_tf_idf_classifiers.py multilabel slovene
```

---

## English dataset
Binary dataset:
```bash
 python run_tf_idf_classifiers.py binary english
```
Multilabel dataset 
```bash
 python run_tf_idf_classifiers.py multilabel english
```
