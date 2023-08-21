import pytest
import torch
from transformers import BertTokenizer

from hipool import curiam_reader


@pytest.fixture
def curiam_sample():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset = curiam_reader.CuriamDataset(
        json_file_path="fixtures/data/curiam_sample.json",
        tokenizer=bert_tokenizer,
        chunk_len=50,
        overlap_len=10)
    return dataset


def test_read_json_len(curiam_sample):
    assert len(curiam_sample.documents) == 2
    assert len(curiam_sample.labels) == 2


def test_read_json_tokens(curiam_sample):
    assert curiam_sample.documents[0][0] == "Justice"
    assert curiam_sample.documents[0][1] == "GORSUCH"


def test_getitem_labels(curiam_sample):
    first_chunk_labels = curiam_sample[0]["targets"][0]
    first_token_labels = first_chunk_labels[0]
    assert torch.equal(first_token_labels, torch.zeros((9)))
