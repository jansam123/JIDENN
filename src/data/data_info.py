import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import pandas as pd


def generate_data_distributions(dataset: tf.data.Dataset,
                                folder: str,
                                var_names: list[str]):

    # create numpy arrays of data, labels and weights from dataset
    data, labels, weights = zip(*dataset.as_numpy_iterator())
    data = np.array(data)
    labels = np.array(labels)
    weights = np.array(weights)

    # create pandas dataframe with data, labels and weights
    df = pd.DataFrame(data=data, columns=var_names)
    # df = pd.DataFrame()
    # for i, var_name in enumerate(var_names):
    #     df[var_name] = data[:, i]
    df['label'] = labels
    df['weight'] = weights
    df['label'] = df['label'].replace({0: 'gluon', 1: 'quark'})

    # create distributions of data, labels and weights
    for var_name in var_names+['label', 'weight']:
        sns.histplot(data=df, x=var_name, hue='label', stat='count')
        plt.savefig(os.path.join(folder, f'{var_name}.png'))
        plt.close()

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # create correlation matrix of data without label
    corr_matrix = df[var_names+['label']].corr()
    fig = plt.figure(figsize=(21, 18))
    # [x0, y0, width, height]
    fig.add_axes([0.2, 0.2, 0.8, 0.8])

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, cmap=cmap, center=0, square=True, linewidths=.5,
                annot=True, fmt='.1f', cbar_kws={'shrink': .8})
    # plt.xticks(rotation=40)
    plt.savefig(os.path.join(folder, 'correlation_matrix.png'))
    plt.close()

