from itertools import chain

import torch
from jaxtyping import jaxtyped
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torcheval.metrics import BinaryF1Score, BinaryPrecision, BinaryRecall
from tqdm import tqdm
from typeguard import typechecked

from hipool.models import SentenceClassificationModel
from hipool.curiam_categories import REDUCED_CATEGORIES

def collate(batch):
    return batch


def collate_sentences(batch):
    batch_input_ids = [torch.tensor(sent["input_ids"], dtype=torch.long) for sent in batch]

    batch_subword_mask = [sent["subword_mask"] for sent in batch]
    batch_labels = [sent["labels"] for sent in batch]
    batch_sent_labels = [sent["sent_labels"] for sent in batch]
    padded_input_ids = pad_sequence(batch_input_ids, batch_first=True).cuda()
    padded_mask = padded_input_ids.not_equal(0).long().cuda()
    padded_token_type_ids = torch.zeros(padded_input_ids.shape, dtype=torch.long, device=torch.device("cuda"))
    return {"input_ids": padded_input_ids,
            "attention_mask": padded_mask,
            "token_type_ids": padded_token_type_ids,
            "subword_mask": batch_subword_mask,
            "labels": batch_labels,
            "sent_labels": batch_sent_labels}


def train_loop(doc_data_loader, sent_model: SentenceClassificationModel,
               optimizer, device, scheduler=None):
    sent_model.train()

    losses = []

    for i, batch_docs in enumerate(tqdm(doc_data_loader)):
        # Batches are usually larger than 1, but we use 1 doc at a time
        doc = batch_docs[0]
        sent_dataset = doc[0]
        chunks = doc[1]

        sent_dataloader = DataLoader(sent_dataset, batch_size=3,
                                     sampler=SequentialSampler(sent_dataset), collate_fn=collate_sentences)

        for i, batch in enumerate(sent_dataloader):
            output = sent_model(ids=batch["input_ids"],
                                mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"])

            targets = torch.cat((batch["sent_labels"]), dim=0).float().cuda()
            # Pick outputs to eval
            # Don't need [cls], [sep], pad tokens, or non-first subwords
            optimizer.zero_grad()
            loss_func = torch.nn.BCEWithLogitsLoss()
            loss = loss_func(output, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            losses.append(loss.detach().cpu())
    return losses

def eval_loop(doc_data_loader, sent_model: SentenceClassificationModel,
              optimizer, device, num_labels):
    sent_model.eval()

    with torch.no_grad():
        metrics = [{"p": BinaryPrecision(device=device),
                    "r": BinaryRecall(device=device),
                    "f": BinaryF1Score(device=device)} for i in range(num_labels)]

        targets_total = []
        for doc_batch_id, batch_docs in enumerate(doc_data_loader):
            doc = batch_docs[0]
            sent_dataset = doc[0]
            chunks = doc[1]
            sent_dataloader = DataLoader(sent_dataset, batch_size=4,
                                         sampler=SequentialSampler(sent_dataset),
                                         collate_fn=collate_sentences)
            output_to_eval = []
            targets_to_eval = []
            for i, batch in enumerate(sent_dataloader):
                targets_to_eval.append(batch["sent_labels"])
                # print("batch ", i, len(sent_dataloader))
                output = sent_model(ids=batch["input_ids"],
                                    mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"])
                output_to_eval.append(output)

            output_to_eval = torch.cat((output_to_eval), dim=0)
            sigmoid_outputs = nn.functional.sigmoid(output_to_eval)
            predictions = (sigmoid_outputs > .5).long().to(device)
            targets_to_eval = list(chain(*targets_to_eval))
            targets = torch.cat((targets_to_eval), dim=0).long().cuda()
            targets_total.append(targets)

            for i in range(num_labels):
                metrics[i]["p"].update(predictions[:, i], targets[:, i])
                metrics[i]["r"].update(predictions[:, i], targets[:, i])
                metrics[i]["f"].update(predictions[:, i], targets[:, i])

        targets_total = torch.cat((targets_total), dim=0)
        sum_amount = torch.sum(targets_total, dim=0)
        print("\tp\tr\tf")
        for i, class_metrics in enumerate(metrics):
            p = class_metrics["p"].compute().item()
            r = class_metrics["r"].compute().item()
            f = class_metrics["f"].compute().item()
            print(f"class {i}\t{p:.4f}\t{r:.4f}\t{f:.4f}\t{torch.sum(targets_total[:, i])}")
