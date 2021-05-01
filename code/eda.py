import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.cluster import SpectralCoclustering


def run_ttests(X, y, equal_var=False):
    """
    Run t-tests over features in a pandas Data.Frame between groups defined in a
    binary label vector, `y`.

    Parameters
    ------------
    dat: {pd.DataFrame } numeric features. Shape {observations} x {featuress}
    y: {vector-like} pd.Series or np.array of binary target labels in {0, 1}. Shape {observations} x 1
    equal_var: {bool} should t-test assume equal variance between the two groups in `y`?

    Returns
    -----------
    significant_features: {list} column names in dat found to have significantly different means
    between the two groups
    """
    signif_features = list()

    # -- get indices of positive/negative class instances
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # -- iterate over features
    for feature in X.columns:

        fvec = X[feature].values

        # -- run t-test
        pval = ttest_ind(fvec[pos_idx]
                         , b=fvec[neg_idx]
                         , axis=0
                         , equal_var=equal_var).pvalue

        if pval <= 0.05:
            signif_features.append(feature)
            print(feature + ' p-value = {0:.4f}'.format(pval))

    return signif_features


def rank_missingness(X, top_n=None, figsize=None):
    """
    Rank the columns of a DataFrame by their rate of missingness. Also provide
    barchart of missingness levels.

    Parameters
    ------------
    X: {array-like} pd.DataFrame or numpy.ndarray. Shape {observations} x {features}
    top_n: {int} display the top-n columns in dat with the highest rate of missing values
    figsize: {list or tuple} of length 2, w x h of resulting plot.

    Returns
    ------------
    missingness: {pd.Series} ordered rankings of features by their missingness rate (which is
                 a proportion in [0, 1])
    """
    missing_dat = X.isnull().sum(axis=0) / X.shape[0]
    missing_dat.sort_values(ascending=False
                            , inplace=True)

    if isinstance(top_n, int):
        top_n = min(top_n, missing_dat.shape[0])
        missing_dat = missing_dat[0:top_n]

    # -- if figsize specified, create barplot out of missingness rates.
    if figsize is not None:
        p = missing_dat.plot(kind='barh', figsize=[6.4, 4.8] if figsize else figsize)
        p.set_xlabel('Missingness %')

    return missing_dat


def find_brutally_categorical_features(X, brutal=20):
    """
    Find brutally categorical features, i.e. categorical features with >= k unique levels.

    Parameters
    ------------
    dat: pd.DataFrame consisting only of categorical features
    brutal: {int > 0} number of unique levels found in a feature before the feature
            is declared "brutally categorical". Default is 20.

    Returns
    ------------
    list of feature names with at least brutal unique levels
    """
    bcat_features = list()

    for feature in X.columns:
        vcounts = X[feature].value_counts()
        if vcounts.shape[0] >= brutal:
            bcat_features.append(feature)

    return bcat_features


def bicluster_correlation_matrix(X, n_clusters=10, figsize=None):
    """
    Group similar variables together by running Spectral coclustering algorithm on a dataset's correlation matrix.
    See https://bit.ly/2QgXZB2 for more details.

    Spectral coclustering finds groups of similar (row, column) subsets where each column can only belong to
    a single bicluster. This is different than "checkerboard" biclustering.

    Parameters
    ------------
    X: {pd.DataFrame} numeric feature data. Shape {observations} x {features}
    n_clusters: {int} number of biclusters to construct
    figsize: {2-tuple of int} pyplot Figure size. Default [10, 6].

    Returns
    ------------
    coclust: {fitted sklearn.cluster.SpectralCoclustering object}
    """

    # -- get estimate of correlation matrix using median-imputed version of data,
    # -- and then downsample to 50k datapoints for speed.
    num_df = X.iloc[np.random.choice(range(X.shape[0])
                                       , size=min(100000, X.shape[0])
                                       , replace=False)]
    cor_mat = num_df.fillna(num_df.median()).corr()

    # -- run coclustering.
    coclust = SpectralCoclustering(n_clusters=n_clusters
                                   , random_state=666)
    coclust.fit(cor_mat)

    # -- re-order correlation matrix by cluster indices.
    biclust_dat = cor_mat.iloc[np.argsort(coclust.row_labels_)]
    biclust_dat = biclust_dat.iloc[:, np.argsort(coclust.column_labels_)]

    # -- display biclustering pattern.
    fig = plt.figure(figsize=figsize if figsize else [10, 10])
    ax = fig.add_subplot(111)
    ax = ax.matshow(biclust_dat
                     , cmap='cool')
    ax.set_title(f'Correlation matrix post-biclustering: {n_clusters} clusters')

    ax.set_yticks(range(biclust_dat.shape[0]))
    ax.set_yticklabels(biclust_dat.index.tolist())

    plt.show()

    return coclust