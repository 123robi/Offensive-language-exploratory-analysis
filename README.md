# Cross-lingual offensive language identification

## Installation Python 3.7

To be able to run this project you need to run following command:
```bash
 pip install -r requirements.txt
```

And then run: 

```bash
 python run_tf_idf.py
```
---
## About
This is a TF-IDF based program that tries to detect offensive language using different models:
* LogisticRegressionModel
* GaussianNaiveBayesModel
* RandomForestClassifierModel
* MultinomialNaiveBayesModel
* BernoulliNaiveBayesModel

Firstly, we prepare our dataset using helper functions which:
* remove stopwords
* remove mentions from the dataset
* remove additional spaces
* keep only characters (remove numbers, punctuations)
* stem words
* tokenization
---
## [Database](https://github.com/t-davidson/hate-speech-and-offensive-language)
We used Twitter database presented in the Paper `Automated Hate Speech Detection and the Problem of Offensive Language`
which divides the data into 3 classes:
* hate speech
* offensive language
* neither

---
# Results
```
LogisticRegressionModel, Accuracy Score: 0.8988567585743107
GaussianNaiveBayesModel, Accuracy Score: 0.7038332212508406
RandomForestClassifierModel, Accuracy Score: 0.901546738399462
MultinomialNaiveBayesModel, Accuracy Score: 0.8480161398789509
BernoulliNaiveBayesModel, Accuracy Score: 0.8796234028244788
```
