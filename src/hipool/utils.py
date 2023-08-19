import time

import numpy as np
import torch
from jaxtyping import Float, jaxtyped
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typeguard import typechecked


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


@jaxtyped
@typechecked
def eval_token_classification(model_output: Tensor[Float, "s c num_labels"],
                              targets: list[Float[Tensor, "s c num_labels"]],
                              overlap_len, device,
                              num_labels):
    """Remove extra token predictions from output and evaluate.

    HiPool takes overlapping chunks as input. For sequence classification, this
    isn't an issue, but for token classification, that means we have multiple
    predictions for some tokens. We remove those my masking out the overlapping
    portion of each chunk, except for the first one in the document, which has
    no overlap.

    All tokens are still accounted for since the removed tokens will still have
    predictions from when they made up the end of the preceding chunk.
    """

    num_chunks = model_output.shape[0]
    chunk_len = model_output.shape[1]
    overlap_mask = torch.zeros((chunk_len), dtype=torch.int)
    overlap_mask[:overlap_len] = 1
    overlap_mask.repeat((num_chunks))
    overlap_mask[:overlap_len] = 0  # There is no overlap for the first chunk

    token_output = model_output.reshape(-1, num_labels)
    # Select only the tokens that aren't overlaps
    deduplicated_output = token_output[overlap_mask == 0]

    # Threshold each label at .5 to get bool predictions then convert to float
    predictions = (deduplicated_output > .5).float()  # noqa F841

    # ---------- Sample implementation of binary multilabel metrics ---------- #
    # Rework this implementation to update on each doc/batch and then print
    # output
    num_labels = 5
    metrics = [{"p": BinaryPrecision(),
                "r": BinaryRecall(),
                "f": BinaryF1Score()} for i in range(num_labels)]

    num_examples = 30
    # Toy data
    pred = (torch.randn((num_examples, num_labels)) > .5).int()
    labels = (torch.randn((num_examples, num_labels)) > .2).int()

    for i in range(num_labels):
        metrics[i]["p"].update(pred[:, i], labels[:, i])
        metrics[i]["r"].update(pred[:, i], labels[:, i])
        metrics[i]["f"].update(pred[:, i], labels[:, i])

    print("\tp\tr\tf")
    for i, class_metrics in enumerate(metrics):
        p = class_metrics["p"].compute().item()
        r = class_metrics["r"].compute().item()
        f = class_metrics["r"].compute().item()
        print(f"class {i}\t{p:.4f}\t{r:.4f}\t{f:.4f}")
