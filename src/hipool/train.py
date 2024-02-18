import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from hipool.models import DocModel, TokenClassificationModel
from hipool.utils import collate_sentences


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
                masked_output = masked_output[batch["subword_mask"][i] == 1]
                output_to_eval.append(masked_output)

            output_to_eval = torch.cat((output_to_eval), dim=0)
            targets = torch.cat((batch["labels"]), dim=0).float().cuda()
            # Pick outputs to eval
            # Don't need [cls], [sep], pad tokens, or non-first subwords
            optimizer.zero_grad()
            loss_func = torch.nn.BCEWithLogitsLoss()
            loss = loss_func(output_to_eval, targets)
            loss.backward(retain_graph=False)
            optimizer.step()
            scheduler.step()

            losses.append(loss.detach().cpu())
    return losses