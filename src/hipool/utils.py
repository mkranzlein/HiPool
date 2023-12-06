from itertools import chain

import numpy as np
import torch
from jaxtyping import jaxtyped
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torcheval.metrics import BinaryF1Score, BinaryPrecision, BinaryRecall
from tqdm import tqdm
from typeguard import typechecked

from hipool.models import DocModel, TokenClassificationModel


def collate_sentences(batch):
    batch_input_ids = [torch.tensor(sent["input_ids"], dtype=torch.long) for sent in batch]

    batch_subword_mask = [sent["subword_mask"] for sent in batch]
    batch_labels = [sent["labels"] for sent in batch]
    padded_input_ids = pad_sequence(batch_input_ids, batch_first=True).cuda()
    padded_mask = padded_input_ids.not_equal(0).long().cuda()
    padded_token_type_ids = torch.zeros(padded_input_ids.shape, dtype=torch.long, device=torch.device("cuda"))
    return {"input_ids": padded_input_ids,
            "attention_mask": padded_mask,
            "token_type_ids": padded_token_type_ids,
            "subword_mask": batch_subword_mask,
            "labels": batch_labels}


def train_loop(doc_data_loader, token_model: TokenClassificationModel,
               optimizer, device, scheduler=None, doc_model: DocModel = None):
    token_model.train()
    if doc_model:
        doc_model.train()
    losses = []

    for i, batch_docs in enumerate(tqdm(doc_data_loader)):
        # Batches are usually larger than 1, but we use 1 doc at a time
        doc = batch_docs[0]
        sent_dataset = doc[0]
        chunks = doc[1]

        sent_dataloader = DataLoader(sent_dataset, batch_size=3,
                                     sampler=SequentialSampler(sent_dataset), collate_fn=collate_sentences)

        for i, batch in enumerate(sent_dataloader):
            if token_model.use_doc_embedding:
                # Get hipool embedding
                doc_embedding = doc_model(chunks)
                output = token_model(ids=batch["input_ids"],
                                     mask=batch["attention_mask"],
                                     token_type_ids=batch["token_type_ids"],
                                     doc_embedding=doc_embedding)
            else:
                output = token_model(ids=batch["input_ids"],
                                     mask=batch["attention_mask"],
                                     token_type_ids=batch["token_type_ids"],)

            output_to_eval = []
            ignoreables = torch.tensor([101, 0, 102]).cuda()
            for i, sent in enumerate(output):
                real_token_mask = torch.isin(elements=batch["input_ids"][i],
                                             test_elements=ignoreables,
                                             invert=True).long()
                masked_output = sent[real_token_mask == 1]
                masked_output = masked_output[torch.tensor(batch["subword_mask"][i]) == 1]
                output_to_eval.append(masked_output)

            output_to_eval = torch.cat((output_to_eval), dim=0)
            targets = torch.cat((batch["labels"]), dim=0).float().cuda()
            # Pick outputs to eval
            # Don't need [cls], [sep], pad tokens, or non-first subwords
            optimizer.zero_grad()
            loss_func = torch.nn.BCEWithLogitsLoss()
            loss = loss_func(output_to_eval, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

            losses.append(loss.detach().cpu())
    return losses


def get_dataset_size(doc_data_loader):
    result = 0
    for batch_docs in doc_data_loader:
        doc = batch_docs[0]
        sent_dataset = doc[0]
        result += len(sent_dataset)
    return result


def split_dataset(size: int, validation_split, seed, shuffle=False):
    indices = list(range(size))
    split = int(np.floor(validation_split * size))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def eval_loop(doc_data_loader, token_model: TokenClassificationModel,
              optimizer, device, num_labels, doc_model: DocModel = None):
    token_model.eval()
    if doc_model:
        doc_model.eval()

    with torch.no_grad():
        metrics = [{"p": BinaryPrecision(device=device),
                    "r": BinaryRecall(device=device),
                    "f": BinaryF1Score(device=device)} for i in range(num_labels)]

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
                targets_to_eval.append(batch["labels"])
                # print("batch ", i, len(sent_dataloader))
                if doc_model:
                    # Get hipool embedding
                    doc_embedding = doc_model(chunks)
                    output = token_model(ids=batch["input_ids"],
                                         mask=batch["attention_mask"],
                                         token_type_ids=batch["token_type_ids"],
                                         doc_embedding=doc_embedding)
                else:
                    output = token_model(ids=batch["input_ids"],
                                         mask=batch["attention_mask"],
                                         token_type_ids=batch["token_type_ids"])

                ignoreables = torch.tensor([101, 0, 102]).cuda()
                for i, sent in enumerate(output):
                    real_token_mask = torch.isin(elements=batch["input_ids"][i],
                                                 test_elements=ignoreables,
                                                 invert=True).long()
                    masked_output = sent[real_token_mask == 1]
                    subword_mask = torch.tensor(batch["subword_mask"][i])
                    masked_output = masked_output[subword_mask == 1]
                    output_to_eval.append(masked_output)

            output_to_eval = torch.cat((output_to_eval), dim=0)
            sigmoid_outputs = nn.functional.sigmoid(output_to_eval)
            predictions = (sigmoid_outputs > .5).long().to(device)
            targets_to_eval = list(chain(*targets_to_eval))
            targets = torch.cat((targets_to_eval), dim=0).long().cuda()

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
