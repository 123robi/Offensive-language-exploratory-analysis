from transformers import BertTokenizer
import tensorflow as tf
import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Load pretrained BertTokenizer
tokenizer = BertTokenizer.from_pretrained('./models/BERT/CroSloEng_FT_Multi_Eng', do_lower_case=True)
device = torch.device("cuda")

# Load training dataset
train_set = pd.read_csv("data/transformed_datasets/train_slo.csv", encoding='utf-8')
X = train_set['text']
y = train_set['subtype']

input_ids = []
attention_masks = []

for text in X:
    encoded_dict = tokenizer.encode_plus(
        text,  # text.
        add_special_tokens=True,  # [CLS] and [SEP] tokens'
        max_length=64,
        pad_to_max_length=True,  # Pad missing tokens with 0s
        return_attention_mask=True,
        return_tensors='pt',  # pytorch tensors.
    )

    input_ids.append(encoded_dict['input_ids'])

    # differentiates padding from non-padding
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y)

dataset = TensorDataset(input_ids, attention_masks, labels)
# Split training set into training and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16

train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = BertForSequenceClassification.from_pretrained(
    "./models/BERT/CroSloEng_FT_Multi_Eng", # Pretrained model path
    num_labels = 5, # multi classification
    output_attentions = False,
    output_hidden_states = False
)



# Use GPU
model.cuda()


# #if layers frozen
# Freezing layers except the classifying layer
# for param in model.bert.parameters():
#     param.requires_grad = False
# optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 5e-5, eps = 1e-8)
# #else
optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)

epochs = 5

total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

total_t0 = time.time()

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # unfreeze layers for futher training
    # if epoch_i == 5:
    #     for param in model.bert.parameters():
    #         param.requires_grad = True
    #     optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Copy to GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear calculated gradients
        model.zero_grad()
        output = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
        loss = output[0]
        logits = output[1]

        # Loss for the batch
        total_train_loss += loss.item()
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    # Batch loss
    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()


    model.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        # Copy to GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)


        with torch.no_grad():
            output = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = output[0]
            logits = output[1]

        total_eval_loss += loss.item()

        # Retrieve data from GPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

import os
output_dir = './models/BERT/CroSloEng_FT_Multi_Slo2/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving model to %s" % output_dir)

model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))



