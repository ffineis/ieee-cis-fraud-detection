import re
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from scipy.sparse import issparse

from eda import rank_missingness, find_brutally_categorical_features
from transformers import *

def find_pipeline_params(p, pattern, verbose=False):
    """
    Find pipeline parameters that match a pattern.

    Parameters
    -----------
    p: {sklearn.pipeline} pipeline object
    pattern: {str} string pattern for matching pipeline parameters
    verbose: {bool} print message for matches?


    Return
    -----------
    matches: {list of str} pipeline parameters matching pattern
    """
    match_params = list()
    for k in p.get_params().keys():
        if re.search(pattern, string=k):
            match_params.append(k)

            if verbose:
                msg = f'parameter matching pattern "{pattern}": {k}'
                print(msg)

    return match_params


def categorical_feature_pipeline(X
                                 , categorical_string_features
                                 , categorical_numeric_features
                                 , binarize=True
                                 , binarize_cutoff=0.5):
    """
    Define a categorical feature processing pipeline.

    Parameters
    ------------
    X: {array-like} pd.DataFrame or numpy.ndarray. Shape {observations} x {features}
    categorical_string_features: {list of str} column names in X representing string-valued
                                 categorical features
    categorical_numeric_features: {list of str} column names in X representing float/int-valued
                                 categorical features
    binarize: {bool} should high-missingness features be binarized with MissingIndicator?
    binarize_cutoff: {float in [0, 1]} threshold above which we create a binary variable
                    for each feature with a missingness rate > binarize_cutoff

    Returns
    ------------
    sklearn.pipeline object
    """
    cat_features = categorical_string_features + categorical_numeric_features

    # -- rank feature missingness.
    cat_missing_dat = rank_missingness(X[cat_features])

    # -- find high-cardinality categorical features
    hi_card_features = find_brutally_categorical_features(X[cat_features])

    # -- separate categorical features into 4 groups for feature preprocessing:
    # -- (string/numeric data type, high/lo-cardinality) groups.
    cat_num_hi_card_features = [x for x in categorical_numeric_features if x in hi_card_features]
    cat_str_hi_card_features = [x for x in categorical_string_features if x in hi_card_features]
    cat_num_lo_card_features = [x for x in categorical_numeric_features if x not in hi_card_features]
    cat_str_lo_card_features = [x for x in categorical_string_features if x not in hi_card_features]

    # -- high-cardinality pipeline = (str-imputer + median-imputer) -> OHE -> SVD.
    hi_card_imputer = ColumnTransformer(
        [('impute_hi_card_cat_num', SimpleImputer(strategy='constant', fill_value=-999), cat_num_hi_card_features)
            ,
         ('impute_hi_card_cat_str', SimpleImputer(strategy='constant', fill_value='NA'), cat_str_hi_card_features)])
    hi_card_pipeline = make_pipeline(hi_card_imputer
                                     , OneHotEncoder(handle_unknown='ignore')
                                     , TruncatedSVD(n_components=100))

    # -- low-cardinality pipeline = (str-imputer + median-imputer) -> OHE.
    lo_card_imputer = ColumnTransformer(
        [('impute_lo_card_cat_num', SimpleImputer(strategy='constant', fill_value=-999), cat_num_lo_card_features)
            ,
         ('impute_lo_card_cat_str', SimpleImputer(strategy='constant', fill_value='NA'), cat_str_lo_card_features)])
    lo_card_pipeline = make_pipeline(lo_card_imputer
                                     , OneHotEncoder(handle_unknown='ignore'))

    # -- Return a column-concatenation of separate categorical feature pipelines
    if not binarize:
        return FeatureUnion([('hi_card', hi_card_pipeline)
                            , ('lo_card', lo_card_pipeline)])

    else:
        # -- append missingness binarizer if specified
        cat_binarize_features = cat_missing_dat[cat_missing_dat > binarize_cutoff].index.tolist()
        cat_binarizer = ColumnTransformer([('missing_cat_binarizer', MissingIndicator(), cat_binarize_features)])

        return FeatureUnion([('hi_card', hi_card_pipeline)
                            , ('lo_card', lo_card_pipeline)
                            , ('missing_cat_binarizer', cat_binarizer)])


def numeric_feature_pipeline(X
                             , numeric_features
                             , binarize=True
                             , binarize_cutoff=0.5):
    """
    Define a numeric feature processing pipeline.

    Parameters
    ------------
    X: {array-like} pd.DataFrame or numpy.ndarray. Shape {observations} x {features}
    numeric_features: {list of str} column names in X representing continuously-valued features
    binarize: {bool} should high-missingness features be binarized with MissingIndicator?
    binarize_cutoff: {float in [0, 1]} threshold above which we create a binary variable
                    for each feature with a missingness rate > binarize_cutoff

    Returns
    ------------
    sklearn.pipeline object
    """
    # -- base of numeric feature processor: imputer + mean, standard deviation scaler
    num_pipeline = make_pipeline(ColumnTransformer([('impute_num', SimpleImputer(strategy='median'), numeric_features)])
                                 , StandardScaler())

    # -- append missingness binarizer if specified
    if binarize:
        num_missing_dat = rank_missingness(X[numeric_features])
        num_binarize_features = num_missing_dat[num_missing_dat > binarize_cutoff].index.tolist()
        num_binarizer = ColumnTransformer([('missing_num_binarizer', MissingIndicator(), num_binarize_features)])

        # -- column-concatenate separate missingness-binarized variables.
        num_pipeline = FeatureUnion([('num_pipeline', num_pipeline)
                                    , ('missing_num_binarizer', num_binarizer)])

    return num_pipeline