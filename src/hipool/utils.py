import time

import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence


def collate(batches):
    # Return batches
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def loss_fun(outputs, targets):
    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, targets)


def evaluate(target, predicted):
    true_label_mask = [1 if (np.argmax(x) - target[i]) == 0
                       else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction - true_prediction
    accuracy = true_prediction / nb_prediction
    return {
        "accuracy": accuracy,
        "nb exemple": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
    }


def train_loop(data_loader, model, optimizer, device, scheduler=None):
    '''optimized function for Hi-BERT'''

    model.train()
    t0 = time.time()
    losses = []

    for batch_idx, batch in enumerate(data_loader):

        ids = [data["ids"] for data in batch]  # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]  # length: 8

        # Here x[0] is used because the label is sequence-level, which
        # makes the label the same for all chunks from a sequence

        # Chunk
        # target_labels = torch.stack([x.squeeze()[0] for x in targets]).float().to(device)

        # Token-level

        padded_labels = pad_sequence(targets).float().to(device)
        padded_labels = padded_labels.permute(1, 0, 2, 3)
        optimizer.zero_grad()

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fun(outputs, padded_labels)
        loss.backward()
        model.float()
        optimizer.step()
        if scheduler:
            scheduler.step()
        losses.append(loss.item())
        if batch_idx % 10 == 0:
            print(
                f"___ batch index = {batch_idx} / {len(data_loader)} ({100*batch_idx / len(data_loader):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")  # noqa E501
            t0 = time.time()

    return losses


def eval_loop(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    for batch_idx, batch in enumerate(data_loader):
        ids = [data["ids"] for data in batch]  # size of 8
        mask = [data["mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["targets"] for data in batch]  # length: 8
        with torch.no_grad():
            target_labels = torch.stack([x[0] for x in targets]).long().to(device)
            outputs, _ = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fun(outputs, target_labels)
            losses.append(loss.item())

        fin_targets.append(target_labels.cpu().detach().numpy())
        fin_outputs.append(torch.softmax(outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses
