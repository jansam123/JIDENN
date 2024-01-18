import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import argparse

from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA, MC_NAMING_SCHEMA

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--load_dir", default=".", type=str,
                    help="Directory to load the metrics from.")


def main(args):
    models = os.listdir(args.load_dir)
    df = []
    for model in models:
        model_df = pd.read_csv(os.path.join(
            args.load_dir, model, 'tech_info.csv'))
        model_df = model_df.T.reset_index(drop=True)
        model_df.columns = model_df.iloc[0]
        model_df = model_df.drop(model_df.index[0]).reset_index(drop=True)
        names = list(model_df.columns)
        values = list(model_df.iloc[0])
        model_df = dict(zip(names, values))
        model_df['model'] = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model
        df.append(model_df)

    df = pd.DataFrame(df)

    df['time'] = df['time'].round(2)
    print(df[['model', 'time']])


if __name__ == '__main__':
    args = parser.parse_args()
    args.load_dir = '/home/jankovys/JIDENN/logs/r22_central_all/evaluation/pythia_nominal-pt-cpu/models'
    main(args)
