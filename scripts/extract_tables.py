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
parser.add_argument("--save_dir", default=".", type=str,
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
        metric_df = pd.read_csv(os.path.join(
            args.load_dir, model, 'overall_metrics.csv'))
        metric_df['model'] = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model
        # join the two dataframes by coulmns
        model_df = pd.DataFrame(model_df, index=[0])
        df.append(pd.concat([model_df, metric_df], axis=1))

    df = pd.concat(df)
    # df = pd.DataFrame(df)
    df = df[['model', 'auc', 'gluon_rejection_at_quark_50wp',
             'num_params', 'time', 'memory']]
    # sort by auc
    df = df.sort_values(by=['auc'], ascending=False)
    # rename columns
    df = df.rename(columns={'model': 'Model', 'num_params':
                   r'\# Params [$10^6$]', 'time': 'Inferece Time [ms]', 'memory': 'GPU Memory [MB]', **METRIC_NAMING_SCHEMA})
    column_format = 'l' + 'c'*(len(df.columns)-1)
    df.to_latex(buf=os.path.join(args.save_dir, 'results.tex'), column_format=column_format, escape=False, index=False,
                label='tab:results',  formatters={**{k: lambda x: f'{x:.4f}' for k in METRIC_NAMING_SCHEMA.values()}, **{r'\# Params [$10^6$]': lambda x: f'{x/1e6:.2f}', 'Inferece Time [ms]': lambda x: f'{x:.2f}', 'GPU Memory [MB]': lambda x: f'{x:.2f}'}}, position='htb')

    print(df)


if __name__ == '__main__':
    args = parser.parse_args()
    args.load_dir = '/home/jankovys/JIDENN/logs/r22_forward_lead_sublead/evaluation/pythia_nominal-eta/models'
    args.save_dir = '/home/jankovys/JIDENN/logs/r22_forward_lead_sublead/evaluation/pythia_nominal-eta'
    main(args)
