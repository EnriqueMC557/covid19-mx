import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sys.path.append("..")

from utils import paths


def plot_single_evaluation_results(data: pd.DataFrame, metric: str = 'accuracy'):
    ax = sns.barplot(data=data.sort_values(by=[metric.lower()], ascending=False), y='model_name', x=metric.lower(), orient='h')
    ax.set_title(f'{metric} results')
    ax.set_xlim(0, 1)
    for label in ax.containers:
        ax.bar_label(label, fmt='%.4f')
    plt.savefig(f'{paths.FIGURES_PATH}/single/{metric.lower()}_results.png', bbox_inches='tight')


def plot_kfold_evaluation_results(data: pd.DataFrame, metric: str = 'accuracy', y_lim=(0, 1), **kwargs):
    sub_results = data.groupby(['model'])[metric].describe()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(17, 5)
    if kwargs.get("plottype", "box") == "violin":
        sns.violinplot(data=data, x='model', y=metric, ax=ax)
    else:
        sns.boxplot(data=data, x='model', y=metric, ax=ax)
    ax.set_ylabel(metric.upper())
    ax.set_ylim(y_lim[0], y_lim[1])

    for x, wt in enumerate(ax.get_xticklabels()):
        median_ = sub_results.loc[wt.get_text(), '50%']
        min_ = sub_results.loc[wt.get_text(), 'min']
        max_ = sub_results.loc[wt.get_text(), 'max']
        q1_ = sub_results.loc[wt.get_text(), '25%']
        q3_ = sub_results.loc[wt.get_text(), '75%']

        ax.text(x, median_, f'{median_:.4f}', horizontalalignment='center',
                color='w', backgroundcolor='k', weight='semibold')
        # ax.text(x, q1_, f'{q1_:.4f}', horizontalalignment='center',
        #         color='w', backgroundcolor='k', weight='semibold')
        # ax.text(x, q3_, f'{q3_:.4f}', horizontalalignment='center',
        #         color='w', backgroundcolor='k', weight='semibold')

    if "figpath" in kwargs:
        plt.savefig(kwargs["figpath"], bbox_inches='tight')
    else:
        plt.savefig(f'{paths.FIGURES_PATH}/kfold/{metric}_results.png', bbox_inches='tight')
    plt.show()

    if "resultspath" in kwargs:
        sub_results.to_csv(kwargs["resultspath"])
    else:
        sub_results.to_csv(f'{paths.RESULTS_PATH}/kfold/{metric}_results.csv')

    return sub_results
