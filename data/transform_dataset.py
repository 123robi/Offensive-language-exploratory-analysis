import csv
import json

fieldnames = ['hatespeech', 'subtype', 'text']

dataset_name = 'whitesupremacy.csv'
folder_path = 'datasets\\'
save_folder_path = 'transformed_datasets\\'

# filepath to dataset
if 'csv' in dataset_name:
    input_file = csv.DictReader(open(folder_path + dataset_name, 'r', encoding="UTF-8"))
elif 'json' in dataset_name:
    input_file = open(folder_path + dataset_name, 'r', encoding="UTF-8")
    if dataset_name == 'CONAN.json':
        input_file = json.load(input_file)['conan']
    else:
        input_file = input_file.readlines()

# filepath to save location
transformed_dataset = open(save_folder_path + dataset_name, 'w', newline='', encoding="UTF-8")
writer = csv.DictWriter(transformed_dataset, fieldnames=fieldnames)
writer.writeheader()

# save conan texts to exclude duplicates
conan_texts = []

for row in input_file:
    if dataset_name == 'whitesupremacy.csv':
        if row['label'] == 'hate':
            writer.writerow({'hatespeech': 1, 'subtype': -1, 'text': row['content']})
        else:
            writer.writerow({'hatespeech': 0, 'subtype': -1, 'text': row['content']})

    elif dataset_name == 'twitter.csv':
        if row['class'] == '2':
            writer.writerow({'hatespeech': 0, 'subtype': -1, 'text': row['tweet']})
        else:
            writer.writerow({'hatespeech': 1, 'subtype': row['class'], 'text': row['tweet']})

    elif dataset_name == 'reddit.csv' or dataset_name == 'gab.csv':
        text_rows = row['text']
        if (row['hate_speech_idx'] != 'n/a'):
            idxs = list(map(int, row['hate_speech_idx'][1:-1].split(", ")))
            for idx in idxs:
                try:
                    text = text_rows.split('\n')[idx - 1]
                    text = ' '.join(text.split()).split('. ')[1:][0]
                except:
                    print(idxs, idx, text)
                writer.writerow({'hatespeech': 1, 'subtype': -1, 'text': text})
    elif dataset_name == 'fox-news.json':
        row = json.loads(row)
        writer.writerow({'hatespeech': row['label'], 'subtype': -1, 'text': row['text']})

    elif dataset_name == 'CONAN.json':
        language = row['cn_id'][0:2]
        if language == 'EN':
            if not row['hateSpeech'] in conan_texts:
                # subtype 2 - islamophobia
                writer.writerow({'hatespeech': 1, 'subtype': 2, 'text': row['hateSpeech']})
                conan_texts.append(row['hateSpeech'])


