import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import os
sns.set_theme(style="darkgrid")
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--files", nargs='*', type=str, help="csv files to load.")
parser.add_argument("--model_names", nargs='*', type=str, help="names of the models.")
parser.add_argument("--x_axis_labels", nargs='*', type=str, help="labels for the x axis.")
parser.add_argument("--save_dir", default=".", type=str, help="Directory to save the plots to.")
parser.add_argument("--take", default=0, type=int, help="Directory to save the plots to.")



def main(args: argparse.Namespace):
    filenames = ['good_logs/comparison/depart/eval/results.csv', 'good_logs/comparison/gated_depart_GEGLU/eval/results.csv', 'good_logs/comparison/gated_depart_ReGLU/eval/results.csv',
                 'good_logs/comparison/part/eval/results.csv', 'good_logs/comparison/highway/eval/results.csv', 'good_logs/comparison/transformer/eval/results.csv', 'good_logs/comparison/bdt/eval/results.csv']
    model_names = ['depart', 'gated_depart_GEGLU', 'gated_depart_ReGLU', 'part', 'highway', 'transformer', 'bdt']
    save_dir = 'good_logs/comparison/figs/'
    take = args.take
    # filenames = args.files
    # x_axis_labels = args.x_axis_labels
    # model_names = args.model_names
    # save_dir = args.save_dir
    # define the labels for the columns in the CSV files

    # load the CSV files into a list of pandas DataFrames
    dataframes = []
    for filename, model_name in zip(filenames, model_names):
        df = pd.read_csv(filename)
        df['model'] = model_name
        if take > 0:
            df = df.iloc[take:]
        dataframes.append(df)

    # concatenate the DataFrames into a single DataFrame
    df = pd.concat(dataframes)

    os.makedirs(save_dir, exist_ok=True)
    for metric in df.columns.drop(['model', 'cut']):
        fig = plt.figure(figsize=(14, 9))
        plot_ = sns.pointplot(x='cut', y=metric, data=df, hue='model', ci=95)
        plt.xlabel('$p_{\mathrm{T}}$ [GeV]')
        plt.ylabel(metric)

        plt.savefig(save_dir + f'{metric}.{take}.jpg')
        plt.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
