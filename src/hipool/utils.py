from itertools import chain

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


def collate(batch):
    return batch


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


def generate_dataset_html(doc_data_loader, token_model, num_labels, tokenizer, device,
                          doc_model=None):
    """Generates color-coded HTML for analyzing model predictions."""
    label_index_to_name = {0: "focal_term", 1: "metalinguistic_cue", 2: "direct_quote", 3: "legal_source"}

    token_model.eval()
    if doc_model:
        doc_model.eval()

    with torch.no_grad():
        for doc_batch_id, batch_docs in enumerate(doc_data_loader):
            doc = batch_docs[0]
            sent_dataset = doc[0]
            chunks = doc[1]
            sent_dataloader = DataLoader(sent_dataset, batch_size=4,
                                         sampler=SequentialSampler(sent_dataset),
                                         collate_fn=collate_sentences)

            doc_input_ids = []
            doc_subword_masks = []
            doc_logits = []
            doc_targets = []
            for batch in sent_dataloader:
                doc_targets.append(batch["labels"])
                if doc_model:
                    # Get hipool embedding
                    doc_embedding = doc_model(chunks)
                    batch_logits = token_model(ids=batch["input_ids"],
                                               mask=batch["attention_mask"],
                                               token_type_ids=batch["token_type_ids"],
                                               doc_embedding=doc_embedding)
                else:
                    batch_logits = token_model(ids=batch["input_ids"],
                                               mask=batch["attention_mask"],
                                               token_type_ids=batch["token_type_ids"])
                doc_input_ids.append(batch["input_ids"])
                doc_subword_masks.append(batch["subword_mask"])
                doc_logits.append(batch_logits)

            # chain lists together
            doc_input_ids = list(chain(*doc_input_ids))
            doc_subword_masks = list(chain(*doc_subword_masks))
            doc_logits = list(chain(*doc_logits))
            doc_targets = list(chain(*doc_targets))

            for k in range(num_labels):
                doc_html = ""
                single_label_logits = [z[:, k] for z in doc_logits]
                single_label_targets = [y[:, k] for y in doc_targets]
                for input_ids, subword_mask, logits, targets in zip(doc_input_ids, doc_subword_masks,
                                                                    single_label_logits, single_label_targets):
                    doc_html += generate_sentence_html(input_ids, subword_mask, logits, targets,
                                                       tokenizer, device)

                with open(f"doc_{doc_batch_id}_{label_index_to_name[k]}.html", "w") as f:
                    f.write(doc_html)


def generate_sentence_html(input_ids, subword_mask,
                           logits, targets, tokenizer, device):
    ignoreables = torch.tensor([101, 0, 102]).to(device)
    real_token_mask = torch.isin(elements=input_ids,
                                 test_elements=ignoreables,
                                 invert=True).long()  # maybe long unnecessary
    cleaned_input_ids = input_ids[real_token_mask == 1]  # Remove [CLS], [SEP], and [PAD]
    original_tokens = " ".join(tokenizer.convert_ids_to_tokens(cleaned_input_ids))
    sent_html = f"<p>{original_tokens}</p>"  # Display tokens of original sentence

    eval_ids = cleaned_input_ids[subword_mask == 1]
    eval_tokens = tokenizer.convert_ids_to_tokens(eval_ids)
    eval_logits = logits[real_token_mask == 1]
    eval_logits = eval_logits[subword_mask == 1]
    scores = nn.functional.sigmoid(eval_logits)
    predictions = (scores > .5).long().to(device)
    sent_html += "<p>"

    for token, score, prediction, target in zip(eval_tokens, scores, predictions, targets):
        if prediction == target:
            if prediction == 1:
                sent_html += f'<span title="{score}" style = "background-color:green">{token} </span>'  # TP
            else:
                sent_html += f'<span title="{score}" style = "background-color:palegreen">{token} </span>'  # TN
        else:
            if prediction == 1:
                sent_html += f'<span title="{score}" style = "background-color:violet">{token} </span>'  # FP
            else:
                sent_html += f'<span title="{score}" style = "background-color:tomato">{token} </span>'  # FN
    sent_html += "</p>"
    return sent_html
