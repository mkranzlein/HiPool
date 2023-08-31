import time

import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score
from torch import nn, Tensor
from typeguard import typechecked


def collate(batches):
    # Return batches
    return [{key: value for key, value in batch.items()} for batch in batches]


def loss_fun(outputs, targets):
    loss = torch.nn.BCEWithLogitsLoss()
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


def train_loop(data_loader, model, optimizer, device, overlap_len, scheduler=None):
    '''optimized function for Hi-BERT'''

    model.train()
    t0 = time.time()
    losses = []

    for batch_idx, batch in enumerate(data_loader):

        ids = [data["input_ids"] for data in batch]  # size of 8
        mask = [data["attention_mask"] for data in batch]
        first_subword_masks = [data["first_subword_mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["labels"] for data in batch]  # length: 8

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        """ Don't include in loss or eval:
        - Predictions for subwords that aren't the first subword of a token
        - [CLS], [SEP], or [PAD]
        - Redundant tokens from overlapping chunks
        """
        outputs_to_eval = []
        for b in range(len(batch)):
            eval_mask = get_eval_mask(ids[b], overlap_len, outputs.shape[1])
            sample_output = outputs[b, :, :, :]
            # TODO: Assert dimensions here
            sample_output = sample_output[eval_mask == 1]
            sample_output = sample_output[torch.tensor(first_subword_masks[b]) == 1]
            outputs_to_eval.append(sample_output)

        outputs_to_eval = torch.cat(outputs_to_eval, dim=0).to(device)
        # TODO: Figure out types
        # outputs_to_eval = (outputs_to_eval > .5).long()
        targets = torch.cat(targets, dim=0).float().to(device)
        print(sum(targets))
        loss = loss_fun(outputs_to_eval, targets)
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
        ids = [data["input_ids"] for data in batch]  # size of 8
        mask = [data["attention_mask"] for data in batch]
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


@jaxtyped
@typechecked
def get_eval_mask(seq_input_ids,  # : Integer[Tensor, "k c"],
                  overlap_len, longest_seq):
    """Create a mask to identify which tokens should be evaluated."""
    # 1 for real tokens, 0 for special tokens
    pad_length = longest_seq - seq_input_ids.shape[0]
    if pad_length != 0:
        input_ids_padding = torch.zeros(pad_length, seq_input_ids.shape[1])
        seq_input_ids = torch.cat((seq_input_ids, input_ids_padding), dim=0)
    real_token_mask = torch.isin(elements=seq_input_ids,
                                 test_elements=torch.tensor([101, 0, 102]),
                                 invert=True).long()

    num_chunks = seq_input_ids.shape[0]
    chunk_len = seq_input_ids.shape[1]
    overlap_mask = torch.zeros((num_chunks, chunk_len), dtype=torch.int)
    overlap_mask[:, 1:overlap_len + 1] = 1
    # Reset first chunk overlap to 0 for each document in the batch
    overlap_mask[0, 1:overlap_len + 1] = 0
    eval_mask = torch.bitwise_and(real_token_mask, ~overlap_mask)
    return eval_mask


@jaxtyped
@typechecked
def eval_token_classification(data_loader,
                              model,
                              device,
                              overlap_len,
                              num_labels):
    """Remove extra token predictions from output then evaluate.

    HiPool takes overlapping chunks as input. For sequence classification, this
    isn't an issue, but for token classification, that means we have multiple
    predictions for some tokens. We remove those my masking out the overlapping
    portion of each chunk, except for the first one in the document, which has
    no overlap.

    All tokens are still accounted for since the removed tokens will still have
    predictions from when they made up the end of the preceding chunk.
    """

    model.eval()
    metrics = [{"p": BinaryPrecision(device=device),
                "r": BinaryRecall(device=device),
                "f": BinaryF1Score(device=device)} for i in range(num_labels)]
    for batch_idx, batch in enumerate(data_loader):

        ids = [data["input_ids"] for data in batch]  # size of 8
        mask = [data["attention_mask"] for data in batch]
        first_subword_masks = [data["first_subword_mask"] for data in batch]
        token_type_ids = [data["token_type_ids"] for data in batch]
        targets = [data["labels"] for data in batch]  # length: 8

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        """ Don't include in loss or eval:
        - Predictions for subwords that aren't the first subword of a token
        - [CLS], [SEP], or [PAD]
        - Redundant tokens from overlapping chunks
        """
        outputs_to_eval = []
        for b in range(len(batch)):
            eval_mask = get_eval_mask(ids[b], overlap_len, outputs.shape[1])
            sample_output = outputs[b, :, :, :]
            # TODO: Assert dimensions here
            sample_output = sample_output[eval_mask == 1]
            sample_output = sample_output[torch.tensor(first_subword_masks[b]) == 1]
            outputs_to_eval.append(sample_output)

        outputs_to_eval = torch.cat(outputs_to_eval, dim=0).to(device)
        # TODO: Figure out types
        # outputs_to_eval = (outputs_to_eval > .5).long()
        targets = torch.cat(targets, dim=0).to(device)
        sigmoid_outputs = nn.functional.sigmoid(outputs_to_eval)
        predictions = (sigmoid_outputs > .5).long().to(device)
        # loss = loss_fun(outputs_to_eval, targets)
        
        for i in range(num_labels):
            metrics[i]["p"].update(predictions[:, i], targets[:, i])
            metrics[i]["r"].update(predictions[:, i], targets[:, i])
            metrics[i]["f"].update(predictions[:, i], targets[:, i])

    print("\tp\tr\tf")
    for i, class_metrics in enumerate(metrics):
        p = class_metrics["p"].compute().item()
        r = class_metrics["r"].compute().item()
        f = class_metrics["f"].compute().item()
        print(f"class {i}\t{p:.4f}\t{r:.4f}\t{f:.4f}")

