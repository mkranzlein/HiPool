# HiPool

A work-in-progress modified implementation of HiPool to support experiments on [CuRIAM](https://arxiv.org/abs/2305.14719).

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
    - [CuRIAM](data/curiam.json): Included with repo
    - IMDB: I think this is the dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
        - I renamed the main csv file to `imdb_sample.csv` and removed most rows for faster debugging.

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
