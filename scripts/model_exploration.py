"""Script for testing dataset loading functionality."""


from hipool.Dataset_Split_Class import DatasetSplit
from hipool.utils import collate

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import BertTokenizer


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset = DatasetSplit(
    tokenizer=bert_tokenizer,
    max_len=1024,
    chunk_len=20,
    overlap_len=10,
    file_location="data/imdb.json")

validation_split = .2
shuffle_dataset = True
random_seed = 42

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

train_data_loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=train_sampler, collate_fn=collate)

valid_data_loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=valid_sampler, collate_fn=collate)

# MK: Nex step is to go through the model code below and figure out what outputs of tokenlevel model look like.
# print('Model building done.')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

# num_training_steps = int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

# if args.level == 'sent':
#     model = Hi_Bert_Classification_Model_GCN(args=args, num_class=dataset.num_class, device=device,
#                                              adj_method=args.adj_method).to(device)
# else:
#     model = Hi_Bert_Classification_Model_GCN_tokenlevel(num_class=dataset.num_class, device=device,
#                                                         adj_method=args.adj_method).to(device)

# optimizer = AdamW(model.parameters(), lr=lr)
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=0,
#                                             num_training_steps=num_training_steps)
# val_losses = []
# batches_losses = []
# val_acc = []
# avg_running_time = []
# for epoch in range(EPOCH):

#     t0 = time.time()
#     print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
#     batches_losses_tmp = train_loop_fun1(train_data_loader, model, optimizer, device)
#     epoch_loss = np.mean(batches_losses_tmp)
#     print("\n ******** Running time this step..", time.time()-t0)
#     avg_running_time.append(time.time()-t0)
#     print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
#     t1 = time.time()
#     output, target, val_losses_tmp = eval_loop_fun1(valid_data_loader, model, device)
#     print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
#     tmp_evaluate = evaluate(target.reshape(-1), output)
#     print(f"=====>\t{tmp_evaluate}")
#     val_acc.append(tmp_evaluate['accuracy'])
#     val_losses.append(val_losses_tmp)
#     batches_losses.append(batches_losses_tmp)
#     print("\t§§ model has been saved §§")

# print("\n\n$$$$ average running time per epoch (sec)..", sum(avg_running_time)/len(avg_running_time))
# # torch.save(model, "models/"+model_dir+"/model_epoch{epoch+1}.pt")
