import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np


def data_distribution(data, folder, names, labels):
    os.makedirs(folder, exist_ok=True)
    for var, name in zip(data, names):
        sns.histplot({'quark (1)':var[np.where(labels == 1)], 'gluon (0)':var[np.where(labels == 0)]}).set(title=name)
        plt.savefig(f'{folder}/{name}.png')
        plt.clf()
        
def generate_data_distributions(datasets, base_folder, size, var_names, datasets_names, func=None):
    for dt, folder in zip(datasets, datasets_names):
        dt = next(dt.unbatch().take(size).batch(size).as_numpy_iterator())
        dt_labels = dt[1]
        vars = func(dt[0]) if func is not None else dt[0]
        dt = np.concatenate([vars, dt[1][:,np.newaxis], dt[2][:, np.newaxis]], axis=1).T
        dt_names = var_names + ["labels", "weights"]
        data_distribution(dt, f'{base_folder}/{folder}', dt_names, dt_labels)