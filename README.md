Modeling Affect Intensity
==============================

Modeling affect intensity expressed in tweets. Semeval task 2018. Authors - Aarsh Patel & Lynn Samson

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── dev            <- Dev data.
    │   ├── train          <- Train data
    │   ├── lexicons       <- Affect/Sentiment Lexicons
    │   └── word2vec       <- Word2Vec/Glove embeddings.
    │
    ├── models             <- Implementation of various models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Relevant literatue for the project
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── utils.             <- Folder contain utility scripts (generating features, preprocessing tweets...)
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── evaluate.py        <- Evaluation script


## Training and evaluating models

```
python evaluate.py --model model_name --features feature_type --metrics metric1, metric2 --optimize True/False
```


## Testing out Baseline Model

```
python evaluate.py --model baseline
```
