"""Trains a model to identify sentences containing metalanguage.

This model is designed for sentence-level classification instead of
token-level classification.
"""

import time

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from hipool import utils
from hipool.curiam_reader import DocDataset
from hipool.models import DocModel, SentenceClassificationModel
from hipool.utils import collate, get_dataset_size
from hipool.sent_model_utils import train_loop, eval_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
bert_model = transformers.BertModel.from_pretrained("bert-base-uncased").cuda()

# Hyperparameters and config
chunk_len = 150
overlap_len = 20
num_labels = 4
TRAIN_BATCH_SIZE = 6  # Number of sentences per batch
EPOCH = 5
hipool_linear_dim = 32
hipool_hidden_dim = 32
hipool_output_dim = 32
lr = 1e-5  # 1e-3
use_hipool = False

doc_dataset = DocDataset(json_file_path="data/curiam.json", tokenizer=bert_tokenizer, bert_model=bert_model,
                         num_labels=num_labels, chunk_len=chunk_len, overlap_len=overlap_len)

print(len(doc_dataset))

train_indices, val_indices = utils.split_dataset(len(doc_dataset), validation_split=.3,
                                                 seed=15, shuffle=True)

train_data_loader = DataLoader(
    Subset(doc_dataset, train_indices[:10]),
    batch_size=1,  # Number of documents ber batch (use 1)
    collate_fn=collate)

valid_data_loader = DataLoader(
    Subset(doc_dataset, val_indices[:]),
    batch_size=1,  # Number of documents ber batch (use 1)
    collate_fn=collate)

print(f"{len(valid_data_loader)} documents in validation set")

sent_model = SentenceClassificationModel(num_labels=num_labels, bert_model=bert_model,
                                         device=device,).to(device)

optimizer = torch.optim.AdamW(sent_model.parameters(), lr=lr)

num_training_steps = int(get_dataset_size(train_data_loader) / TRAIN_BATCH_SIZE * EPOCH)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=10,
                                            num_training_steps=num_training_steps)

for epoch in range(EPOCH):

    t0 = time.time()
    batches_losses_tmp = train_loop(train_data_loader, sent_model, optimizer, device, scheduler)
    epoch_loss = np.mean(batches_losses_tmp)
    print(f"Epoch {epoch} average loss: {epoch_loss} ({time.time() - t0} sec)")
    eval_loop(valid_data_loader, sent_model, optimizer, device, num_labels)

torch.save(sent_model, "models/curiam/sentence_level_model_nohipool.pt")
