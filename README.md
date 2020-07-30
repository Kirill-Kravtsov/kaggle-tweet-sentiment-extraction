# A Pytorch pipeline for [Tweet Sentiment Extraction Kaggle competition](https://www.kaggle.com/c/tweet-sentiment-extraction)

## Introduction
This is a Pytorch training pipeline for a text span selection task. It also uses the [Catalyst](https://github.com/catalyst-team/catalyst) deep learning framework.

## Installation
1. You need to have [Anaconda](https://anaconda.org) installed
2. Clone the repo
```bash
git clone https://github.com/Kirill-Kravtsov/kaggle-tweet-sentiment-extraction
```
3. Create and activate provided Anaconda enviroment
```bash
conda env create -f tweet_env.yml
conda activate tweet_env
```
4. Download competition data and put in `data` dir in root of the project
5. Create folds by running
```bash
python create_folds.py
```

## Project structure:
```bash
├── configs
│   ├── best_bertweet.yml
│   ├── best_roberta.yml
│   ├── experiments
│   └── optimization
├── create_folds.py
├── data
├── logs
├── scripts
├── src
│   ├── callbacks.py
│   ├── collators.py
│   ├── datasets.py
│   ├── data_utils.py
│   ├── hooks.py
│   ├── losses.py
│   ├── optimize_experiment.py
│   ├── tokenization.py
│   ├── train.py
│   ├── transformer_models.py
│   └── utils.py
└── tweet_env.yml
```

## Running pipeline
To train tha basic Roberta and BERTweet models run:
```bash
python train.py --cv --config ../configs/best_roberta.yml
python train.py --cv --config ../configs/best_bertweet.yml
```
Note: the code is supposed to work with one gpu, so if you have multi-gpu system do not forget to specify `CUDA_VISIBLE_DEVICE` variable, e.g.:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --cv --config ../configs/best_roberta.yml
```
