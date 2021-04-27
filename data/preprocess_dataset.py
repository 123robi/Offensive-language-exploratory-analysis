import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split

fieldnames = ['hatespeech', 'subtype', 'text']

dataset_name = 'merged_dataset.csv'
folder_path = 'transformed_datasets\\'
save_folder_path = 'transformed_datasets\\'

input_file = csv.DictReader(open(folder_path + dataset_name, encoding='utf-8'))

train_dataset = open(save_folder_path + 'train.csv', 'w', newline='', encoding='utf-8')
test_dataset = open(save_folder_path + 'test.csv', 'w', newline='', encoding='utf-8')

train_w = csv.DictWriter(train_dataset, fieldnames=fieldnames)
test_w = csv.DictWriter(test_dataset, fieldnames=fieldnames)
train_w.writeheader()
test_w.writeheader()

rows = list(input_file)
random.shuffle(rows)

hate = []
nohate = []

for row in rows:
    row = dict(row)
    if (row['hatespeech'] == '1'):
        hate.append(row)
    else:
        nohate.append(row)

print("Hate rows: " + str(len(hate)) + "\nNo hate rows: " + str(len(nohate)))

hate_balanced = np.random.choice(hate, len(nohate), replace=False)
print("Balanced hate rows: " + str(len(hate_balanced)))


train_hate, test_hate = train_test_split(hate_balanced, test_size=0.2)
train_nohate, test_nohate = train_test_split(nohate, test_size=0.2)

train_set = np.concatenate([train_hate, train_nohate])
test_set = np.concatenate([test_hate, test_nohate])
random.shuffle(train_set)
random.shuffle(test_set)

for row in train_set:
    row = dict(row)
    train_w.writerow({'hatespeech': row['hatespeech'], 'subtype': row['subtype'], 'text': row['text']})

for row in test_set:
    row = dict(row)
    test_w.writerow({'hatespeech': row['hatespeech'], 'subtype': row['subtype'], 'text': row['text']})