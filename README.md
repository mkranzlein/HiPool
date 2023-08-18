# HiPool

A work-in-progress modified implementation of HiPool to support experiments on [CuRIAM](https://arxiv.org/abs/2305.14719).

This is **not the original repo for HiPool** and I am not an author on the HiPool paper. Please see that repo [here](https://github.com/irenezihuili/hipool).

## Links
- HiPool paper: https://aclanthology.org/2023.acl-short.16/
- Original HiPool repo: https://github.com/irenezihuili/hipool
- HiPool's implementation is based on: https://github.com/helmy-elrais/RoBERT_Recurrence_over_BERT/blob/master/train.ipynb

## Setup
1. Create conda environment.
    ```
    conda env create -f environment.yml
    conda activate hipool
    ```
2. Install hipool locally.
    ```
    pip install --upgrade build
    pip install -e .
    ```
3. Download datasets.
    - CuRIAM: [Included]((data/curiam.json)) with repo.
    - IMDB: I think [this](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) is the dataset. 
        - I renamed the main csv file to `imdb_sample.csv` and removed most rows for faster debugging, since this dataset is not important for what I'm experimenting with.

## Misc

[work-in-progress]

This repo uses [jaxtyping](https://github.com/google/jaxtyping) and [typeguard](https://typeguard.readthedocs.io/) to enforce correct tensor dimensions at runtime. If you see an unfamiliar type annotation or decorators like in the example the below, it's for type checking.

```
@jaxtyped
@typechecked
def some_function(x: Float[torch.Tensor, "10, 768"]):
    pass
```

I recommend taking a look at the [jaxtyping docs](https://docs.kidger.site/jaxtyping/).

## TODOs
- Some long documents are too big for GPU vram right now
- Batching right now should allow for single documents, but worth testing
- Chunking logic could be more clear
- Last chunk of each doc might be improperly ignored?
- HiPool gcn code needs type annotations and improved clarity
- Eval script needs to be adapted to token level classification
    - Overlaps in particular will need to be accounted for. Can start with full first chunk, then start with non-overlapping part of next chunk

## Cite HiPool
```
@inproceedings{li2023hipool,
  title={HiPool: Modeling Long Documents Using Graph Neural Networks},
  author={Li, Irene and Feng, Aosong and Radev, Dragomir and Ying, Rex},
  booktitle={Proceedings of the Association for Computational Linguistics (ACL)},
  year={2023},
  url={https://aclanthology.org/2023.acl-short.16/}
}
```
