import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import AdamW

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import get_linear_schedule_with_warmup
import time

from utils import collate, rnn_eval_loop_fun1, rnn_train_loop_fun1, evaluate
from Custom_Dataset_Class import ConsumerComplaintsDataset
from BERT_Hierarchical import BERT_Hierarchical_BERT_Model
import warnings
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

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./us-consumer-finance-complaints/consumer_complaints.csv")

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

train_raw = df[df.consumer_complaint_narrative.notnull()]
print('Number of training sentences with complain narrative not null: {:,}\n'.format(train_raw.shape[0]))

# Display 10 random rows from the data.
print(train_raw.sample(10))

train_raw['len_txt'] = train_raw.consumer_complaint_narrative.apply(lambda x: len(x.split()))
train_raw.describe()

# Select only the row with number of words greater than 250:
train_raw = train_raw[train_raw.len_txt > 249]
print(train_raw.shape)

# Select only the column 'consumer_complaint_narrative' and 'product'
train_raw = train_raw[['consumer_complaint_narrative', 'product']]
train_raw.reset_index(inplace=True, drop=True)
train_raw.head()

# Group similar products
train_raw.at[train_raw['product'] == 'Credit reporting', 'product'] = 'Credit reporting, credit repair services, or other personal consumer reports'  # noqa E501
train_raw.at[train_raw['product'] == 'Credit card', 'product'] = 'Credit card or prepaid card'
train_raw.at[train_raw['product'] == 'Prepaid card', 'product'] = 'Credit card or prepaid card'
train_raw.at[train_raw['product'] == 'Payday loan', 'product'] = 'Payday loan, title loan, or personal loan'
train_raw.at[train_raw['product'] == 'Virtual currency', 'product'] = 'Money transfer, virtual currency, or money service'  # noqa E501
print(train_raw.head())

# all the different classes
for a in np.unique(train_raw['product']):
    print(a)

train_raw = train_raw.rename(columns={'consumer_complaint_narrative': 'text', 'product': 'label'})
train_raw.head()

print('Data check done.')

TRAIN_BATCH_SIZE = 3
EPOCH = 15
validation_split = .2
shuffle_dataset = True
random_seed = 42
MIN_LEN = 249
MAX_LEN = 100000
CHUNK_LEN = 200
OVERLAP_LEN = 50
# MAX_LEN=10000000
# MAX_SIZE_DATASET=1000

print('Loading BERT tokenizer...')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataset = ConsumerComplaintsDataset(
    tokenizer=bert_tokenizer,
    min_len=MIN_LEN,
    max_len=MAX_LEN,
    chunk_len=CHUNK_LEN,
    # max_size_dataset=MAX_SIZE_DATASET,
    overlap_len=OVERLAP_LEN)

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
    sampler=train_sampler,
    collate_fn=collate)

valid_data_loader = DataLoader(
    dataset,
    batch_size=TRAIN_BATCH_SIZE,
    sampler=valid_sampler,
    collate_fn=collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# lr=3e-5
lr = 2e-5

num_training_steps = int(len(dataset) / TRAIN_BATCH_SIZE * EPOCH)

pooling_method = "mean"

model_hierarchical = BERT_Hierarchical_BERT_Model(device=device, pooling_method=pooling_method).to(device)
optimizer = AdamW(model_hierarchical.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=num_training_steps)
val_losses = []
batches_losses = []
val_acc = []
for epoch in range(EPOCH):
    t0 = time.time()
    print(f"\n=============== EPOCH {epoch+1} / {EPOCH} ===============\n")
    batches_losses_tmp = rnn_train_loop_fun1(train_data_loader, model_hierarchical, optimizer, device)
    epoch_loss = np.mean(batches_losses_tmp)
    print(f"\n*** avg_loss : {epoch_loss:.2f}, time : ~{(time.time()-t0)//60} min ({time.time()-t0:.2f} sec) ***\n")
    t1 = time.time()
    output, target, val_losses_tmp = rnn_eval_loop_fun1(valid_data_loader, model_hierarchical, device)
    print(f"==> evaluation : avg_loss = {np.mean(val_losses_tmp):.2f}, time : {time.time()-t1:.2f} sec\n")
    tmp_evaluate = evaluate(target.reshape(-1), output)
    print(f"=====>\t{tmp_evaluate}")
    val_acc.append(tmp_evaluate['accuracy'])
    val_losses.append(val_losses_tmp)
    batches_losses.append(batches_losses_tmp)
    print(f"\t§§ the Hierarchical {pooling_method} pooling model has been saved §§")
    torch.save(model_hierarchical, f"model_hierarchical/{pooling_method}_pooling/model_{pooling_method}_pooling_epoch{epoch+1}.pt")  # noqa E501
