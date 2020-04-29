import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from scipy.sparse import issparse


class UncorrelatedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom feature selector to drop highly-correlated features.
    This should technically inherit from a SelectorMixin class, as this is a an implementation
    of a feature selector not a feature transformer, but the TransformerMixin api is easier to work with.

    Args:
        drop_threshold: (float) absolute value of correlation coefficient, beyond which
                        one of (feature_i, feature_j) will be dropped should they be found to be highly correlated.
        dsample: (float or int) either fraction of dataset or integer number of rows to sample in order to compute
                correlation matrix.
    """
    def __init__(self, cor_threshold=0.95, dsample=50000, seed=None):
        self.cor_threshold = cor_threshold
        self.dsample = dsample
        self._is_df = False
        self._mask = None
        self._seed = seed

    def fit(self, X, y=None):
        """
        Sample data to create a correlation matrix and find features in
        X that are highly correlated with others.

        Parameters
        -----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
        """
        n, p = X.shape

        # -- determine downsampling rate for computing correlation matrix.
        if isinstance(self.dsample, float) and self.dsample > 0 and self.dsample <= 1:
            dsample = self.dsample * n

        elif self.dsample > 1:
            dsample = self.dsample

        else:
            dsample = n

        # -- create downsampling index vector.
        dsample = min(dsample, n)

        if self._seed:
            np.random.seed(self._seed)

        self._dsample_vec = np.random.choice(range(n)
                                             , size=dsample
                                             , replace=False)

        # -- workflow for dataframe input.
        if isinstance(X, pd.DataFrame):
            cor_mat_abs = X.iloc[self._dsample_vec].corr().abs().values
            self._is_df = True

        # -- workflow for numpy array input.
        else:
            # -- np.corrcoef by default thinkns of rows as variables... and cols as observations(?)
            take_cor = X[self._dsample_vec, :] if not issparse(X) else X[self._dsample_vec, :].todense()
            cor_mat_abs = np.abs(np.corrcoef(take_cor
                                             , rowvar=False))
            del take_cor

        cor_upper = np.triu(cor_mat_abs, k=1)

        # -- Identify multicollinear features and drop them from X.
        hi_cor_features = [j for j in range(p) if np.any(cor_upper[:, j] > self.cor_threshold)]
        self._mask = np.sort(np.array(list(set(range(p)) - set(hi_cor_features))))

        return self

    def transform(self, X, y=None):
        """
        Extract non-correlated features from X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            New data. Must have the same number of columns as the data used to fit the transformer.

        Returns
        -------
        X_new : {same data type as X}, shape (n_samples, n_components)
        """
        if not isinstance(self._mask, np.ndarray):
            raise NotFittedError()

        if self._is_df:
            return X.iloc[:, self._mask]

        else:
            return X[:, self._mask]


class NonDegenerateFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom feature selector to drop degenerate (i.e. very low-variance) features by identifying
    features that take on a single value at least 100% x ecdf_threshold of the time.

    Parameters
    -----------
    ecdf_threshold: {float in [0, 1]} proportion time a feature spends in a single value
                    before the feature is withheld from feature set.
    """
    def __init__(self, ecdf_threshold=0.95):
        ecdf_threshold = 1. if not ecdf_threshold else ecdf_threshold

        if (ecdf_threshold < 0.) or (ecdf_threshold > 1.):
            raise ValueError('ecdf_threshold must lie within [0, 1]')

        self.ecdf_threshold = ecdf_threshold
        self._is_df = None

    def fit(self, X, y=None):
        """
        Identify features that take on a single value at least 100% x ecdf_threshold of the time.

        Parameters
        -----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
        """
        n, p = X.shape
        self._is_df = isinstance(X, pd.DataFrame)

        # -- workflow for pandas dataframe vs. numpy array.
        # -- find features taking on a single value > ecdf_threshold% of the time.
        degen_features = list()
        if self._is_df:
            degen_features = X.apply(lambda z: z.value_counts(normalize=True
                                                              , ascending=False).max() >= self.ecdf_threshold
                                     , axis=0).index.tolist()

        else:
            sp = issparse(X)

            for j in range(p):
                max_occ = np.max(np.unique(X[:, j] if not sp else np.ravel(X[:, j].todense())
                                           , return_counts=True)[1] / n)
                if max_occ >= self.ecdf_threshold:
                    degen_features.append(j)

        self._mask = np.sort(np.array(list(set(range(p)) - set(degen_features))))

        return self

    def transform(self, X, y=None):
        """
        Extract the non-degenerate features from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data. Must have the same number of columns as the data used to fit the transformer.

        Returns
        -------
        X_new : {same data type as X}, shape (n_samples, n_components)
        """
        if not isinstance(self._mask, np.ndarray):
            raise NotFittedError()

        if self._is_df:
            return X.iloc[:, self._mask]

        else:
            return X[:, self._mask]