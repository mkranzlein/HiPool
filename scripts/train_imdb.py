import argparse
import os
import time
import warnings

from hipool.Bert_Classification import Hi_Bert_Classification_Model_GCN, Hi_Bert_Classification_Model_GCN_tokenlevel
from hipool.Dataset_Split_Class import DatasetSplit
from hipool.utils import collate
from hipool.utils import train_loop_fun1, evaluate, eval_loop_fun1

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, BertTokenizer
from transformers import get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--dataset', type=str, default='complaints',
                    help='choose from [complaints, imdb]')
parser.add_argument('--lstm_dim', type=int, default=128, help='Hidden dim for entering graph.')
parser.add_argument('--hid_dim', type=int, default=32, help='Hidden dim for graph models.')
parser.add_argument('--sentlen', type=int, default=20, help='Sentence length.')
parser.add_argument('--epoch', type=int, default=10, help='Number of epoch.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size.')
parser.add_argument('--graph_type', type=str, default='graphsage', help='Graph encoder type: gcn, gat, graphsage, randomwalk,linear')  # noqa E501
parser.add_argument('--adj_method', type=str, default='path_graph',
                    help='choose from [fc,dense_gnm_random_graph,erdos_renyi_graph,binomial_graph,path_graph,complete]')
parser.add_argument('--level', type=str, default='sent', help='level: sent or tok')
args = parser.parse_args()

model_dir = args.dataset
if not os.path.exists("models/"+model_dir):
    os.mkdir("models/"+model_dir)
    print('Making model dir...')

TRAIN_BATCH_SIZE = args.batch_size
EPOCH = args.epoch
validation_split = .2
shuffle_dataset = True
random_seed = 42
MAX_LEN = 1024
GROUP_NUM = 10
'group_num 50, simple model 82%'

# CHUNK_LEN=200
CHUNK_LEN = args.sentlen  # sentence-level
OVERLAP_LEN = int(args.sentlen/2)

lr = 2e-5  # 1e-3

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print('Loading data...', args.dataset)
dataset = DatasetSplit(
    tokenizer=bert_tokenizer,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    overlap_len=OVERLAP_LEN,
    # max_size_dataset=MAX_SIZE_DATASET,
    # file_location='./IMDB',
    file_location=args.dataset)

# Creating data indices for training and validation splits:
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
    batch_size=TRAIN_BATCH_SIZE,
    sampler=train_sampler, collate_fn=collate)

valid_data_loader = DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler, collate_fn=collate)

print('Model building done.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_training_steps = int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

if args.level == 'sent':
    model = Hi_Bert_Classification_Model_GCN(args=args, num_class=dataset.num_class, device=device,
                                             adj_method=args.adj_method).to(device)
else:
    model = Hi_Bert_Classification_Model_GCN_tokenlevel(num_class=dataset.num_class, device=device,
                                                        adj_method=args.adj_method).to(device)

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)
val_losses = []
batches_losses = []
val_acc = []
avg_running_time = []
for epoch in range(EPOCH):

    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp = train_loop_fun1(train_data_loader, model, optimizer, device)
    epoch_loss = np.mean(batches_losses_tmp)
    print("\n ******** Running time this step..", time.time()-t0)
    avg_running_time.append(time.time()-t0)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1 = time.time()
    output, target, val_losses_tmp = eval_loop_fun1(valid_data_loader, model, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate = evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print("\t§§ model has been saved §§")

print("\n\n$$$$ average running time per epoch (sec)..", sum(avg_running_time)/len(avg_running_time))
# torch.save(model, "models/"+model_dir+"/model_epoch{epoch+1}.pt")