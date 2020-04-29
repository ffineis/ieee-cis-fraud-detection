# IEEE-CIS Fraud Detection Kaggle competition code, illustration of massive Sklearn Pipelines

Build a one-stop Sklearn data processing Pipeline and fit a benchmark LASSO or massive ExtraTrees classifier to detect fraudlent credit card transactions for the [IEEE-CIS Fraud Detection Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/).

### Get the data
Log into Kaggle (or set up the kaggle python CLI) and download the competition data from https://www.kaggle.com/c/ieee-fraud-detection/data and unzip the contents of the downloaded .zip file.

### Train a model

Fit the benchmark LASSO model
```
python3 run_pipelines.py --train ../data/train/train_transactions.csv --benchmark
```

Fit the ExtraTrees model
```
python3 run_pipelines.py --train ../data/train/train_transaction.csv
```

### Obtain test set predictions
```
python3 run_pipelines.py --test ../data/test/test_transactions.csv
```
