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
    logdir = 'good_logs/comparison_small/'
    model_names = os.listdir(logdir)
    if 'figs' in model_names:
        model_names.remove('figs')

    result_csvs = []
    usable_models = []
    for model in model_names:
        csv_file = logdir+model+'/eval/results.csv'
        if os.path.isfile(csv_file):
            result_csvs.append(csv_file)
            usable_models.append(model)

    print(result_csvs)
    save_dir = logdir+'figs/'
    os.makedirs(save_dir, exist_ok=True)
    take = args.take

    # load the CSV files into a list of pandas DataFrames
    dataframes = []
    for filename, model_name in zip(result_csvs, usable_models):
        df = pd.read_csv(filename)
        df['model'] = model_name
        if take > 0:
            df = df.iloc[take:]
        dataframes.append(df)

    # concatenate the DataFrames into a single DataFrame
    df = pd.concat(dataframes)
    print(df)

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
