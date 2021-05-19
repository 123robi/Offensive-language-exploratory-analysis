import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification


test_set = pd.read_csv("data/transformed_datasets/nova24_multi.csv", encoding='utf-8')
X = test_set['comment']
y = test_set['type']
input_ids = []
attention_masks = []

tokenizer = XLMRobertaTokenizer.from_pretrained('./models/XLM/XLMRoBERTa', do_lower_case=True)
device = torch.device("cuda")

labels = []
for i in range(len(X)):
    if not pd.isnull(X[i]):
        encoded_dict = tokenizer.encode_plus(
            X[i],  # text.
            add_special_tokens=True,  # [CLS] and [SEP] tokens'
            max_length=64,
            pad_to_max_length=True, # Pad missing tokens with 0s
            return_attention_mask=True,
            return_tensors='pt',  # pytorch tensors.
        )
        labels.append(y[i])
        input_ids.append(encoded_dict['input_ids'])

        # differentiates padding from non-padding
        attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

batch_size = 1
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model = XLMRobertaForSequenceClassification.from_pretrained('./models/XLM/XLMRoBERTa')

model.to(device)
model.eval()

# Tracking variables
predictions, true_labels = [], []
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        # Get predictionss
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Retrieve data from GPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Save predictions
    predictions.append(logits)
    true_labels.append(label_ids)

# Combine the results across all batches.
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Check if predictions are correct
acc = np.sum(flat_predictions == flat_true_labels) / len(flat_predictions)
print(flat_predictions)
print(flat_true_labels)
print(acc)
