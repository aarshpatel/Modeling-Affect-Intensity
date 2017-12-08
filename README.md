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


## Baseline Model 

We used lexicon features in addition to glove embeddings as features to a SVR model as our baseline model. In order to run the baseline model, run the following command:

```
python evaluate.py --model baseline
```

Results:

| Emotion  | CV Pearson Correlation Score  |
| ------------- | ------------- |
| Anger  | 0.66019879032626883  |
| Fear  | 0.67025025487576528  |
| Joy  | 0.68789131129906456  |
| Sadness  | 0.66106523454135879  |
| AVG | 0.66985139776061442 |

## Best Performing Model

A simple feedforward neural network that uses word2vec embeddings and lexicon features as input into the model. 

Results:
| Emotion  | CV Pearson Correlation Score  |
| ------------- | ------------- |
| Anger  | 0.66019879032626883  |
| Fear  | 0.67025025487576528  |
| Joy  | 0.68789131129906456  |
| Sadness  | 0.66106523454135879  |
| AVG | 0.66985139776061442 |


## Experiments:
TODO

## Poster
![Alt text](https://raw.github.com/aarshpatel/Modeling-Affect-Intensity/poster/Affect_Intensity_Poster.png)



