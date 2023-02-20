import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import logging
from typing import Union, List
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, f_classif


def tf_dataset_to_pandas(dataset: tf.data.Dataset, var_names: List[str]) -> pd.DataFrame:
    # create numpy arrays of data, labels and weights from dataset
    data, labels, weights = zip(*dataset.as_numpy_iterator())
    data = np.array(data)
    labels = np.array(labels)
    weights = np.array(weights)

    # create pandas dataframe with data, labels and weights
    df = pd.DataFrame(data=data, columns=var_names)
    df['label'] = labels
    df['weight'] = weights
    return df


def generate_data_distributions(df: pd.DataFrame,
                                folder: str,
                                color_column: str = 'named_label') -> None:
    var_names = list(df.columns)
    corr_matrix = df.corr()
    # create distributions of data, labels and weights
    for var_name in var_names+['label', 'weight']:
        try:
            sns.histplot(data=df, x=var_name, hue=color_column, stat='count')
        except TypeError:
            # logging.warning(f'Could not plot {var_name}, skipping')
            small_df = df[[var_name, color_column]]
            rows = len(df.index)
            small_df = small_df.explode(var_name, ignore_index=True)
            small_df = small_df.sample(n=rows).reset_index(drop=True)
            try:
                sns.histplot(data=small_df, x=var_name, hue=color_column, stat='count')
            except TypeError:
                small_df = small_df.explode(var_name, ignore_index=True)
                small_df = small_df.sample(n=rows).reset_index(drop=True)
                print(small_df)
                sns.histplot(data=small_df, x=var_name, hue=color_column, stat='count')

        plt.savefig(os.path.join(folder, f'{var_name}.png'))
        plt.close('all')

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # create correlation matrix of data without label
    fig = plt.figure(figsize=(21, 18))
    # [x0, y0, width, height]
    fig.add_axes([0.2, 0.2, 0.8, 0.8])

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, linewidths=.5,
                annot=True, fmt='.1f', cbar_kws={'shrink': .8})
    # plt.xticks(rotation=40)
    plt.savefig(os.path.join(folder, 'correlation_matrix.png'))
    plt.close('all')


def plot_feature_importance(df: pd.DataFrame, fig_path: str, score_name: str = 'score', variable_name: str = 'variable'):
    feature_scores = df.sort_values(score_name, ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(10, 15))
    # [x0, y0, width, height]
    ax = fig.add_axes([0.33, 0.05, 0.6, 0.92])
    sns.barplot(x=score_name, y=variable_name, data=feature_scores)
    ax.set_ylabel(ylabel="")
    ax.set_xlabel(xlabel="Score")
    plt.savefig(fig_path)
    plt.close('all')


def feature_importance(df: pd.DataFrame,
                       folder: str,
                       k: Union[int, None] = None):
    X = df.drop(['label', 'weight'], axis=1)
    y = df['label']
    k = len(list(X.columns)) if k is None else k
    for score_name, score_func in zip(['linear', 'mutual'], [f_classif, mutual_info_classif]):
        bestfeatures = SelectKBest(score_func=score_func, k=k)
        fit = bestfeatures.fit(X, y)
        feature_scores = pd.DataFrame({'score': fit.scores_, 'variable': X.columns})
        plot_feature_importance(feature_scores, os.path.join(folder, f'feature_{score_name}.png'))
