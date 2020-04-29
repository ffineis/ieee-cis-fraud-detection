class DataConfig(object):
    """
    Data configuration object for IEEE fraud detection model.
    Stores data processing pipeline settings.
    """
    def __init__(self, target='isFraud', exclude=['TransactionID']
                 , hi_missingness_cutoff=0.5, hi_cardinality_cutoff=6
                 , degenerate_ecdf_cutoff=0.95, multicollinear_cutoff=0.95):
        """
        Set hyperparameters for data processing pipelines.

        Parameters
        -----------
        target: {str} name of target variable
        exclude: {list of str} names of columns that should be excluded from feature set
        hi_missingness_cutoff: {float in [0, 1]} features missing > this fraction of values are considered "highly missing"
        hi_cardinality_cutoff: {int > 0} categorical features with at least this many levels are considered "high cardinality."
        degenerate_ecdf_cutoff: {float in [0, 1]} features taking on a single value more than this % of the time
                                will be considered statistically degenerate variables
        multicollinear_cutoff: {float in [0, 1]} pairs of features with |correlation| at least >= this amount
                               will be considered multicollinear features.
        """
        self.target = target
        self.exclude = list(exclude) if not isinstance(exclude, list) else exclude
        self.hi_cardinality_cutoff = hi_cardinality_cutoff
        self.hi_missingness_cutoff = hi_missingness_cutoff
        self.degenerate_ecdf_cutoff = degenerate_ecdf_cutoff
        self.multicollinear_cutoff = multicollinear_cutoff