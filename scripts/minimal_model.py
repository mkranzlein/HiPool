import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score
from transformers import AdamW, BertTokenizerFast
import json
transformers.DataCollatorForTokenClassification

training_sentences = [
    "the dog jumped over the cat .",
    "cats are cool .",
    "the ocean contains much water .",
    "the sky is blue ."
]
training_labels = [
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0]
]

device = torch.device('cuda')
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)


class CuriamDataset(torch.utils.data.Dataset):
    def __init__(self, json_file_path: str, tokenizer: BertTokenizerFast):
        self.read_json(json_file_path)

    def read_json(self, json_file_path: str) -> list:
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.labels = []
        self.sentences = []

        for raw_document in raw_data:
            sentences = [[token["text"].lower() for token in sentence["tokens"]]
                         for sentence in raw_document["sentences"][:]]

            for sentence in sentences:
                self.sentences.append(sentence)
            # Get labels for actual tokens
            for sentence in raw_document["sentences"][:]:
                sentence_labels = []
                for token in sentence["tokens"]:
                    token_label = 0
                    if "annotations" in token:
                        for annotation in token["annotations"]:
                            annotation_category = annotation["category"]
                            if annotation_category in ["METALINGUISTIC CUE"]:
                                token_label = 1
                    sentence_labels.append(token_label)
                self.labels.append(sentence_labels)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx):

        result = tokenize_and_mask_labels({"tokens": self.sentences[idx], "labels": self.labels[idx]}, bert_tokenizer)
        return result


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        result = tokenize_and_mask_labels({"tokens": self.sentences[idx], "labels": self.labels[idx]}, bert_tokenizer)
        return result


def get_masked_wordpiece_labels(labels: list, word_ids: list) -> list:
    """Returns masked labels for wordpiece tokens.

    The first subword of each token retains the original token label and
    remaining subwords for that token are set to -100.

    Special tokens like CLS and SEP also get a label of -100.

    Subwords with value of -100 will not be included in loss calculation.
    """

    masked_labels = []
    current_word = None
    for word_id in word_ids:
        # Special tokens (CLS and SEP) don't have a word_id
        if word_id is None:
            masked_labels.append(-100)
        # Start of a new word
        elif word_id != current_word:
            current_word = word_id
            label = labels[word_id]
            masked_labels.append(label)
        # Non-first subword of token
        else:
            masked_labels.append(-100)

    return masked_labels


def tokenize_and_mask_labels(examples, tokenizer: BertTokenizerFast):
    """Tokenizes examples and mask associated labels to accomodate wordpiece."""

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True,
                                 is_split_into_words=True)
    token_labels = examples["labels"]
    word_ids = tokenized_inputs.word_ids()
    tokenized_inputs["labels"] = get_masked_wordpiece_labels(token_labels, word_ids)
    return tokenized_inputs


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased").to(device)
        self.linear = torch.nn.Linear(768, 2).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_token_embeddings = []

        results = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        batch_token_embeddings = results["last_hidden_state"]
        batch_model_output = []

        for sequence_token_embedding in batch_token_embeddings:
            l2_output = self.linear(sequence_token_embedding)
            batch_model_output.append(l2_output)
        batch_model_output = torch.stack(batch_model_output, dim=0)
        return batch_model_output


collator = transformers.DataCollatorForTokenClassification(bert_tokenizer)

# training_dataset = CustomDataset(training_sentences, training_labels)
curiam_dataset = CuriamDataset("data/curiam.json", bert_tokenizer)

train_dataloader = DataLoader(
    curiam_dataset,
    batch_size=30,
    collate_fn=collator
)
model = TokenModel()


loss_func = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-4)

metrics = {"p": BinaryPrecision(device=device),
           "r": BinaryRecall(device=device),
           "f": BinaryF1Score(device=device)}

model.train()
num_epochs = 500
for epoch in range(num_epochs):
    for batch in train_dataloader:
        ids = batch["input_ids"].cuda()  # size of 8
        mask = batch["attention_mask"].cuda()
        token_type_ids = batch["token_type_ids"].cuda()
        targets = batch["labels"].cuda().long()  # length: 8.

        outputs = model(ids, mask, token_type_ids)
        softmax = torch.nn.Softmax(dim=2)
        #outputs = softmax(outputs)
        optimizer.zero_grad()
        targets = targets.reshape(-1)
        loss = loss_func(outputs.reshape(-1, 2).float(), targets)
        print(loss.item())
        loss.backward()
        model.float()
        optimizer.step()
        
        # TODO: calc training f1 metrics for sanity check
