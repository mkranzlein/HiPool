
from itertools import chain

import torch
from jaxtyping import jaxtyped
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from torcheval.metrics import BinaryF1Score, BinaryPrecision, BinaryRecall
from typeguard import typechecked

from hipool.models import DocModel, TokenClassificationModel
from hipool.utils import collate_sentences
def eval_loop(doc_data_loader, token_model: TokenClassificationModel,
              device, num_labels, doc_model: DocModel = None):
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
                    subword_mask = batch["subword_mask"][i]
                    masked_output = masked_output[subword_mask == 1]
                    output_to_eval.append(masked_output)

            targets_to_eval = list(chain(*targets_to_eval))

            output_to_eval = torch.cat((output_to_eval), dim=0)
            sigmoid_outputs = nn.functional.sigmoid(output_to_eval)
            predictions = (sigmoid_outputs > .5).long().to(device)

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

def eval_sentence_metalanguage(doc_data_loader, token_model: TokenClassificationModel,
                               optimizer, device, num_labels, doc_model: DocModel = None):
    """High-level evaluation of whether sentences contain any metalanguage."""
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
            sent_labels = []
            for i, batch in enumerate(sent_dataloader):
                for sent in batch["labels"]:
                    pos_token_count = torch.sum(sent, dim=0)
                    sent_label = (pos_token_count >= 1).float()
                    sent_labels.append(sent_label.unsqueeze(0))
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
                    subword_mask = batch["subword_mask"][i]
                    masked_output = masked_output[subword_mask == 1]
                    # Get sentence-level prediction: 1 if model predicts any tokens in sentence as being metalinguistic 0 otherwise
                    output = torch.nn.functional.sigmoid(masked_output)
                    pos_token_prediction_count = torch.sum((output >= .5), dim=0)
                    sentence_prediction = (pos_token_prediction_count >= 1).float()
                    output_to_eval.append(sentence_prediction.unsqueeze(0))

            predictions = torch.cat((output_to_eval), dim=0)
            targets = torch.cat((sent_labels), dim=0).long().cuda()

            for i in range(num_labels):
                metrics[i]["p"].update(predictions[:, i], targets[:, i])
                metrics[i]["r"].update(predictions[:, i], targets[:, i])
                metrics[i]["f"].update(predictions[:, i], targets[:, i])

        print("\tp\tr\tf")
        print("Sentence-level metrics")
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