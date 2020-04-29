from pandas import DataFrame, melt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_precision_recall_curve
sns.set(style='dark')


def plot_gcv_performance(cv, figsize=[12, 6]):
    """
    Create a barplot summarizing gridsearched cross-validation hold out set performance.
    Each individual bar represents a specific model parameter setting's performance metric value.

    Parameters
    ------------
    cv: fitted sklearn.model_selection.GridSearchCV object
    figsize: {2-tuple} width, height parameters of

    Returns
    ------------
    matplotlib.figure.Figure
    """
    # -- build gcv DataFrame
    n_settings = len(cv.cv_results_.get('params'))
    cv_df = DataFrame({'params': [str(cv.cv_results_.get('params')[i]) for i in range(n_settings)]})

    for metric in cv.scoring:
        cv_df[f'mean_{metric}'] = cv.cv_results_.get(f'mean_test_{metric}')

    # -- unpivot data into ggplot-plottable format.
    cv_df = melt(cv_df
                 , id_vars=['params']
                 , var_name='metric'
                 , value_name='y')

    # -- render barplot.
    fig, ax = plt.subplots(figsize=figsize)
    sns.catplot(x='params'
                , y='y'
                , hue='metric'
                , data=cv_df
                , height=6
                , kind='bar'
                , palette='muted'
                , ax=ax)
    ax.set_xticklabels(ax.get_xticklabels()
                       , rotation=45)
    ax.set_title('Cross-validation performance')
    plt.close(2)

    return ax.figure
