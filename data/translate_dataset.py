from google_trans_new import google_translator
import csv
from multiprocessing.dummy import Pool as ThreadPool
import time


import numpy as np

fieldnames = ['hatespeech', 'subtype', 'text']

translator = google_translator()

dataset_name = 'merge.csv'
folder_path = 'transformed_datasets\\'
save_folder_path = 'transformed_datasets\\'


input_file = csv.DictReader(open(folder_path + dataset_name, 'r', encoding="UTF-8-sig"))

# filepath to save location
transformed_dataset = open(save_folder_path + 'slovene2_' + dataset_name, 'w', newline='', encoding="UTF-8")
writer = csv.DictWriter(transformed_dataset, fieldnames=fieldnames)
writer.writeheader()
texts = []
subtypes = []
hatespeech = []
# count = 0
# for row in input_file:
#     row = dict(row)
#     text = translator.translate(row['text'], lang_src='en', lang_tgt='sl')
#     print(count, row['text'], text)
#     count += 1
#     writer.writerow({'hatespeech': row['hatespeech'], 'subtype': row['subtype'], 'text': text})

count = 0
rows = []
for row in input_file:
    row = dict(row)
    rows.append(row)
for i in range(0,len(rows)):
    row = rows[i]
    texts.append(row['text'])
    subtypes.append(row['subtype'])
    hatespeech.append(row['hatespeech'])
pool = ThreadPool(32) # Threads

def request(text):
    lang = "sl"
    t = google_translator(timeout=5)
    translate_text = t.translate(text.strip(), lang)
    return translate_text

if __name__ == "__main__" :
    time1 = time.time()

    try:
      results = pool.map(request, texts)
    except Exception as e:
      raise e
    pool.close()
    pool.join()

    time2 = time.time()
    print(results)
    print("Translating %s sentences, a total of %s s"%(len(texts),time2 - time1))
    for i in range(len(results)):
        writer.writerow({'hatespeech': hatespeech[i], 'subtype': subtypes[i], 'text': results[i]})