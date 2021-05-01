import argparse
import logging
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import sys

from config import DataConfig
from evaluation import *
from feature_pipelines import *


# -- download data from https://www.kaggle.com/c/ieee-fraud-detection/data...
# -- built CLI parser.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--train'
                    , type=str
                    , default=os.path.join(os.getenv('HOME'), 'projects', 'ieee-cis-fraud', 'data', 'train', 'train_transactions.csv')
                    , required=True
                    , help='Filepath of IEEE fraud detection training data')
parser.add_argument('--test'
                    , type=str
                    , default=None
                    , required=False
                    , help='Filepath of IEEE fraud detection test set data')
parser.add_argument('-b'
                    , '--benchmark'
                    , action='store_true'
                    , help='Train and cross validate a benchmark LASSO model.')


if __name__ == '__main__':

    # -- parse CLI args
    args = parser.parse_args()
    data_config = DataConfig()

    # -- set up filepaths and filenames.
    output_dir = os.path.join('..', 'data', 'output')
    file_prefix = 'benchmark_' if args.benchmark else ''
    pipe_fname = '{0}feature_pipeline.pkl'.format(file_prefix)
    est_fname = '{0}estimator.pkl'.format(file_prefix)

    # -- configure logger
    msg = '%(asctime)s ---- %(message)s' if not args.benchmark else '(Benchmark model) %(asctime)s ---- %(message)s'
    logging.basicConfig(level=logging.DEBUG
                        , format=msg
                        , handlers=[logging.StreamHandler()]
                        , datefmt='%m/%d/%Y %I:%M:%S')

    # -- begin test set workflow...
    if args.test:
        preds_fname = '{0}test_set_predictions.csv'.format(file_prefix)
        feature_pipeline = joblib.load(os.path.join(output_dir, pipe_fname))
        model = joblib.load(os.path.join(output_dir, est_fname))

        # -- load training data, data processing configuration.
        logging.info(f'Loading test dataset: {args.test}')
        df = pd.read_csv(args.test)
        logging.info(f'Successfully loaded test dataset, shape: ({df.shape[0]}, {df.shape[1]})')

        # -- run test set through feature processing pipeline.
        x_test = feature_pipeline.transform(df)
        n_test, p_test = x_test.shape
        logging.info(f'x_test shape: ({n_test}, {p_test})')

        # -- obtain test set output, save.
        pos_class_idx = np.where(model.classes_ == 1)[0][0]
        pred_df = pd.DataFrame({'TransactionID': df['TransactionID']
                                , 'fraud_probability': model.predict_proba(x_test)[:, pos_class_idx]
                                , 'is_fraud': model.predict(x_test)})
        pred_df.to_csv(preds_fname
                       , index=False)
        logging.info(f'Successfully obtained test set predictions, shape: ({pred_df.shape[0]}, {pred_df.shape[1]})')
        pred_fraud_rate = 100 * pred_df['is_fraud'].mean()
        logging.info('{:.3f}% of test set transactions have been predicted to be fraudulent'.format(pred_fraud_rate))

        # -- exit from main program.
        sys.exit(0)

    # -- begin training set workflow...
    # -- load training data, data processing configuration
    logging.info(f'Loading training dataset: {args.train}')
    df = pd.read_csv(args.train)
    n, p = df.shape
    fraud_rate = 100 * df[data_config.target].mean()

    logging.info(f'Successfully loaded training dataset, shape: ({n}, {p})')
    logging.info('{:.3f}% of training transactions are fraudulent'.format(fraud_rate))

    # ---------------------------------------------------------- #
    # Identify core sets of features:
    # - numeric features
    # - string-valued categorical features
    # - numeric-valued categorical features
    # ---------------------------------------------------------- #

    # -- identify categorical features with string data type.
    cat_str_features = ['ProductCD', 'P_emaildomain', 'R_emaildomain', 'card4', 'card6'] + ['M' + str(i) for i in
                                                                                            range(1, 10)]
    # -- identify categorical features with integer/float data type.
    cat_num_features = ['addr1', 'addr2', 'card1', 'card2', 'card3', 'card5']

    # -- assemble all categorical features.
    cat_features = cat_str_features + cat_num_features

    # -- assemble numeric features.
    num_features = list()
    for feature in df.columns.tolist():
        if feature in cat_features or feature in data_config.exclude or feature == data_config.target:
            continue

        num_features.append(feature)

    # ---------------------------------------------------------- #
    # Create feature processing pipelines
    # ---------------------------------------------------------- #
    num_pipeline = numeric_feature_pipeline(df
                                            , numeric_features=num_features
                                            , binarize_cutoff=data_config.hi_missingness_cutoff)

    cat_pipeline = categorical_feature_pipeline(df
                                                , categorical_string_features=cat_str_features
                                                , categorical_numeric_features=cat_num_features
                                                , hi_cardinality_cutoff=data_config.hi_cardinality_cutoff
                                                , binarize_cutoff=data_config.hi_missingness_cutoff)

    # -- if specified, define benchmark model using just numeric features
    if args.benchmark:
        feature_pipeline = make_pipeline(num_pipeline
                                         , UncorrelatedFeatureSelector()
                                         , NonDegenerateFeatureSelector(ecdf_threshold=data_config.degenerate_ecdf_cutoff))

    # -- otherwise, use categorical *and* numeric processing pipelines.
    else:
        feature_pipeline = Pipeline(steps=[('preprocess', FeatureUnion([('categorical', cat_pipeline)
                                                                         , ('numeric', num_pipeline)]))
            , ('decorrelate', UncorrelatedFeatureSelector())
            , ('nondegenerate', NonDegenerateFeatureSelector())])

    # ---------------------------------------------------------- #
    # Test/train split
    # ---------------------------------------------------------- #
    logging.info('Splitting data into training, hold-out sets...')
    x_train, x_test, y_train, y_test = train_test_split(df.drop(data_config.target, axis=1)
                                                        , df[data_config.target].values
                                                        , test_size=0.2
                                                        , random_state=666)
    logging.info(f'x_train: ({x_train.shape[0]}, {x_train.shape[1]}); y_train: ({len(y_train)},)')
    logging.info(f'x_test: ({x_test.shape[0]}, {x_test.shape[1]}); y_test: ({len(y_test)},)')

    # ---------------------------------------------------------- #
    # Fit processing pipeline, model pipeline
    # ---------------------------------------------------------- #
    logging.info('Fitting preprocessing pipeline and transforming x_train...')
    x_train = feature_pipeline.fit_transform(x_train)
    n_train, p_train = x_train.shape
    logging.info(f'x_train shape: ({n_train}, {p_train})')

    # -- save fitted feature pipeline.
    joblib.dump(feature_pipeline
                , filename=os.path.join(output_dir, pipe_fname))

    # -- set grid-search cross-validation parameters.
    track_metrics = ['f1', 'roc_auc', 'recall', 'precision', 'accuracy']
    refit = 'f1'
    cv = 3
    n_jobs = 2

    # -- configure benchmark model: LASSO logistic regression.
    if args.benchmark:
        model = LogisticRegression(penalty='l1'
                                   , solver='liblinear'
                                   , class_weight='balanced')
        param_grid = dict(C=[0.1, 1, 10])

    # -- configure full-spec model: ExtraTrees model.
    else:
        model = ExtraTreesClassifier(class_weight='balanced')
        param_grid = dict(max_features=[np.int(np.sqrt(p_train)), p_train]
                          , n_estimators=[50]) #, 100])

    # -- run benchmark model grid search.
    # -- Note: since classes are so unbalanced, accuracy is not a super helpful scoring metric.
    gcv = GridSearchCV(model
                       , param_grid=param_grid
                       , cv=cv
                       , n_jobs=n_jobs
                       , scoring=track_metrics
                       , refit=refit
                       , verbose=10)

    # -- run grid search job. Ensure that targets are integers (by rounding).
    logging.info(f'Begin grid-search cross-validation...')
    gcv.fit(x_train
            , y=y_train)

    # -- extract best estimator.
    logging.info(f'Best parameter settings: {gcv.best_params_}')
    model = gcv.best_estimator_

    # -- save best estimator.
    joblib.dump(model
                , filename=os.path.join(output_dir, est_fname))

    # ---------------------------------------------------------- #
    # Transform hold-out set, get hold-out set predictions
    # ---------------------------------------------------------- #
    x_test = feature_pipeline.transform(x_test)
    test_pred_probs = model.predict_proba(x_test)[:, 1]
    threshold = 0.5
    test_preds = (test_pred_probs > threshold).astype(int)

    # ---------------------------------------------------------- #
    # Evaluate performance on hold-out set.
    # ---------------------------------------------------------- #

    # -- grid search hold-out set performance.
    gcv_plot = plot_gcv_performance(gcv)
    plot_fname = '{0}gcv_performance.png'.format(file_prefix)
    logging.info(f'Saving gridsearch performance plot to {plot_fname}')
    gcv_plot.savefig(os.path.join(output_dir, plot_fname))

    # -- hold-out set precision-recall curve.
    fig, ax = plt.subplots(figsize=[10, 6])
    plot_precision_recall_curve(gcv
                                , X=x_test
                                , y=y_test
                                , ax=ax)
    ax.set_title('Precision-recall curve')
    plot_fname = '{0}pr_curve.png'.format(file_prefix)
    logging.info(f'Saving Hold-out-set precision-recall curve to {plot_fname}')
    ax.figure.savefig(os.path.join(output_dir, plot_fname))

    # -- compute hold-out set metrics.
    test_acc = 100 * np.mean(test_preds == y_test)
    test_tp = np.intersect1d(np.where(test_preds == 1), np.where(y_test == 1))
    test_precision = len(test_tp) / np.sum(test_preds)
    test_recall = len(test_tp) / np.sum(y_test)
    test_roc_auc = roc_auc_score(y_test
                                 ,  y_score = test_pred_probs)

    logging.info('Hold-out set accuracy: {:.3f}%'.format(test_acc))
    logging.info('Hold-out set precision: {:.3f}%'.format(test_precision))
    logging.info('Hold-out set recall: {:.3f}%'.format(test_recall))
    logging.info('Hold-out set f1: {:.3f}%'.format(2 * test_precision * test_recall / (test_precision + test_recall)))
    logging.info('Hold-out set ROC AUC: {:.3f}%'.format(test_roc_auc))
