import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, f_classif


def tf_dataset_to_pandas(dataset: tf.data.Dataset, var_names: list[str]) -> pd.DataFrame:
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
                                folder: str,):
    var_names = list(df.columns)
    corr_matrix = df.corr()
    # create distributions of data, labels and weights
    for var_name in var_names+['label', 'weight']:
        sns.histplot(data=df, x=var_name, hue='named_label', stat='count')
        plt.savefig(os.path.join(folder, f'{var_name}.png'))
        plt.close()

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
    plt.close()

def feature_importance(df: pd.DataFrame, 
                 folder: str,
                 k:int | None = None):
    X = df.drop(['label', 'weight'], axis=1)
    y = df['label']
    k = len(list(X.columns)) if k is None else k
    for score_name, score_func in zip(['linear', 'mutual'], [f_classif, mutual_info_classif]):
        bestfeatures = SelectKBest(score_func=score_func, k=k)
        fit = bestfeatures.fit(X,y)
        feature_scores = pd.DataFrame({'score':fit.scores_, 'variable':X.columns}).sort_values('score', ascending=False).reset_index(drop=True)
        fig = plt.figure(figsize=(10, 15))
        # [x0, y0, width, height]
        ax = fig.add_axes([0.33, 0.05, 0.6, 0.92])
        sns.barplot(x="score", y="variable", data=feature_scores)
        ax.set_ylabel(ylabel="")
        ax.set_xlabel(xlabel="Score")
        plt.savefig(os.path.join(folder, f'feature_{score_name}.png'))
        plt.close()
    

