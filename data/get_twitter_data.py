import twython
import tweepy
import csv
import numpy as np
CONSUMER_KEY = "z4PPh9nCywX2bFS624wCT5QzX"
CONSUMER_SECRET = "JIALmWdnOgt4RW68g8oy1hjudmga0rVfsEtadYaJBekOF6CqWI"
OAUTH_TOKEN = "4372916193-Imx6FkEKBU0mysKsPJCBtztqH25EYskAQA7mvG4"
OAUTH_TOKEN_SECRET = "rqS3y8uy5XCgydyIZkyKAMRmIMuiu9xLkz9BUCcvP2lZp"

# twitter = twython.Twython(
#     CONSUMER_KEY, CONSUMER_SECRET,
#     OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

input_file = csv.DictReader(open("C:\\Users\\edice\\Downloads\\IMSyPP_SI_anotacije_evaluation-clarin.csv", 'r', encoding="UTF-8-sig"))

dataset_name = 'clarin_eval.csv'
folder_path = 'transformed_datasets\\'
save_folder_path = 'transformed_datasets\\'

transformed_dataset = open(save_folder_path + dataset_name, 'w', newline='', encoding="UTF-8")
transformed_dataset_rest = open(save_folder_path + 'clarin_all_eval.csv', 'w', newline='', encoding="UTF-8")
fieldnames = ['hatespeech', 'subtype', 'text']
writer = csv.DictWriter(transformed_dataset, fieldnames=fieldnames)
writer.writeheader()

writer_rest = csv.DictWriter(transformed_dataset_rest, fieldnames=fieldnames)
writer_rest.writeheader()
print("Starting")
count = 0


def lookup_tweets(tweet_IDs, subtypes, api):
    full_tweets = []
    full_subtypes = []
    tweet_count = len(tweet_IDs)
    print(tweet_count/100 +1 )
    count = 0
    try:
        for i in range(int((tweet_count / 100) + 1)):
            # Catch the last group if it is less than 100 tweets
            end_loc = min((i + 1) * 100, tweet_count)
            print(len(tweet_IDs[i * 100:end_loc]))
            if (len(tweet_IDs[i * 100:end_loc]) > 0):
                full_tweets.extend(
                    api.statuses_lookup(id_=tweet_IDs[i * 100:end_loc])
                )
            print(len(full_tweets))
            # count += 1
            # if count == 2:

        return full_tweets
    except tweepy.TweepError:
        print('Something went wrong, quitting...')
        return full_tweets

rows = []
subtypes = []
for row in input_file:
    row = dict(row)
    #ID = ''.join(ch for ch in string.printable if row['ID'].isalnum())
    rows.append(row['ID'])
    #print(str(row['ID']))
    if row['tarča'] != '':
        subtypes.append(row['tarča'][0:2])
    else:
        subtypes.append(row['tarča'])
    # try:
    #     tweet = api.get_status(id)
    #     print(tweet.text)
    #     # if tarca == '':
    #     #     writer.writerow({'hatespeech': 0, 'subtype': 0, 'text': tweet['text']})
    #     # if tarca == '3':
    #     #     writer.writerow({'hatespeech': 1, 'subtype': 4, 'text': tweet['text']})
    #     # if tarca == '6':
    #     #     writer.writerow({'hatespeech': 1, 'subtype': 2, 'text': tweet['text']})
    #     # if tarca == '7':
    #     #     writer.writerow({'hatespeech': 1, 'subtype': 3, 'text': tweet['text']})
    #     # if tarca == '1':
    #     #     writer.writerow({'hatespeech': 1, 'subtype': 1, 'text': tweet['text']})
    #     #
    #     # writer_rest.writerow({'hatespeech': 1, 'subtype': tarca, 'text': tweet['text']})
    #     #
    #     # count += 1
    #     # if count > 5:
    #     #     break
    # except:
    #     print("error")
results = lookup_tweets(rows, subtypes, api)
print(results)
for row in results:
    text = row.text
    id = row.id_str
    tarca = subtypes[rows.index(id)]
    if tarca == '':
        writer.writerow({'hatespeech': 0, 'subtype': 0, 'text': text})
    if tarca == '3 ':
        writer.writerow({'hatespeech': 1, 'subtype': 4, 'text': text})
    if tarca == '6 ':
        writer.writerow({'hatespeech': 1, 'subtype': 2, 'text': text})
    if tarca == '7 ':
        writer.writerow({'hatespeech': 1, 'subtype': 3, 'text': text})
    if tarca == '1 ':
        writer.writerow({'hatespeech': 1, 'subtype': 1, 'text': text})

    writer_rest.writerow({'hatespeech': 1, 'subtype': tarca, 'text': text})