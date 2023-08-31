"""Haphazard hacking trying to understand the internals of these models."""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, BertTokenizerFast, get_linear_schedule_with_warmup

from hipool.curiam_reader import CuriamDataset
from hipool.imdb_reader import IMDBDataset
from hipool.models import TokenClassificationModel
from hipool.utils import collate, train_loop, eval_token_classification

bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
is_curiam = True

chunk_len = 50
overlap_len = 20
num_labels = 3
if is_curiam:
    dataset = CuriamDataset(
        json_file_path="data/curiam.json",
        tokenizer=bert_tokenizer,
        num_labels=num_labels,
        chunk_len=chunk_len,
        overlap_len=overlap_len)
else:
    dataset = IMDBDataset(file_path="data/imdb_sample.csv",
                          tokenizer=bert_tokenizer,
                          max_len=1024,
                          chunk_len=chunk_len,
                          overlap_len=overlap_len)

asdf = dataset[0]
print()
validation_split = .2
shuffle_dataset = True
random_seed = 28

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

batch_size = 2

train_data_loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=train_sampler, collate_fn=collate)

valid_data_loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=valid_sampler, collate_fn=collate)

# MK: Next step is to go through the model code below and figure out what outputs of tokenlevel model look like.
# print('Model building done.')

TRAIN_BATCH_SIZE = 2
EPOCH = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_training_steps = int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)


chunk_model = False

model = TokenClassificationModel(args="", num_labels=num_labels, chunk_len=chunk_len, device=device).to(device)
# else:
#     model = TokenLevelModel(num_class=dataset.num_class, device=device).to(device)


lr = 1e-3  # 1e-3
optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=5,
#                                             num_training_steps=num_training_steps)
val_losses = []
batches_losses = []
val_acc = []
avg_running_time = []
for epoch in range(EPOCH):

    t0 = time.time()
    batches_losses_tmp = train_loop(train_data_loader, model, optimizer, device, overlap_len)
    epoch_loss = np.mean(batches_losses_tmp)
    print(f"Epoch {epoch} average loss: {epoch_loss} ({time.time() - t0} sec)")
    eval_token_classification(valid_data_loader, model, device, overlap_len, num_labels)