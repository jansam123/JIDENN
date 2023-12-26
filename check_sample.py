import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import atlasify
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to save the dataset.")
parser.add_argument("--save_path", type=str,
                    help="Path where to save the plots.")
parser.add_argument("--dataset_type", type=str, default='dev',
                    help="Type of dataset to load. One of ['train', 'dev', 'test'].")
parser.add_argument("--phys_object", type=str, default='jets',
                    help="Physics object to plot.")
parser.add_argument("--jz_description", type=str, default='/home/jankovys/JIDENN/data/sample_description.csv',
                    help="Path to the jz_description.csv file.")

X_LABELS = {'pt': r"$p_{\mathrm{T}}$ [MeV]", 'eta': r"$\eta$",
            'phi': r"$\phi$", 'e': r"$E$ [MeV]", 'm': r"$m$ [MeV]"}


def sns_plot_phys(df, save_path, variable, object_name):
    x_name = X_LABELS[variable]
    df = df.rename(columns={variable: x_name})
    ax = sns.histplot(data=df, x=x_name, weights="weight", log_scale=(
        False, True), bins=100, stat="count", palette='Set1', element="step", fill=False, legend=False)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)

    plt.xlabel(x_name, horizontalalignment='right', x=1.0, fontsize=15)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0, fontsize=15)
    subtext = '13 TeV\n'
    subtext += r'anti-$k_{\mathrm{T}}$, $R = 0.4$ UFO CSSK jets' + \
        '\n' if object_name == 'jets' else ''
    subtext += r'photons'+'\n' if object_name == 'photons' else ''
    subtext += r'muons'+'\n' if object_name == 'muons' else ''
    atlasify.atlasify(
        axes=ax,
        subtext=subtext,
        atlas='Simulation Internal',
        font_size=15,
        sub_font_size=15,
        label_font_size=15,
    )
    plt.savefig(f"{save_path}/{object_name}_{variable}.png",
                bbox_inches='tight')
    plt.savefig(f"{save_path}/{object_name}_{variable}.pdf",
                dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def write_weights(cross_section: float = 1., filt_eff: float = 1., lumi: float = 1., norm: float = 1.):
    @tf.function
    def _calculate_weights(data):
        data = data.copy()
        w = tf.cast(data['weight_mc'], tf.float64)
        cast_lumi = tf.cast(lumi, tf.float64)
        cast_cross_section = tf.cast(cross_section, tf.float64)
        cast_filt_eff = tf.cast(filt_eff, tf.float64)
        cast_norm = tf.cast(norm, tf.float64)
        weight = w[0] * cast_lumi * \
            cast_cross_section * cast_filt_eff / cast_norm
        data['weight'] = weight
        return data
    return _calculate_weights


def load_dataframe(path, vars, cross_section, filt_eff, lumi, norm_name):
    from jidenn.data.JIDENNDataset import JIDENNDataset
    ds = JIDENNDataset.load(path)
    norm = tf.constant(ds.metadata[norm_name])
    ds = ds.remap_data(write_weights(cross_section, filt_eff, lumi, norm))
    vars += ['weight']
    ds = ds.remap_data(lambda x: {var: x[var] for var in vars})
    print(f'Converting to pandas...')
    ds = ds.to_pandas()
    print(f'Converting to pandas done.')
    return ds


def load_all_jzs(path, dataset_type, variables, jz_description_file, phys_object):
    variables = [f'{phys_object}_{var}' for var in variables]
    jzs = os.listdir(path)
    jz_nums = [int(jz.replace('JZ', '')) for jz in jzs]
    jz_description = pd.read_csv(jz_description_file)
    lumi = 139_000

    df = []
    for jz in jz_nums:
        print(f'Loading JZ{jz}')
        file = os.path.join(path, f'JZ{jz}', dataset_type)
        filt_eff = jz_description[jz_description['JZ']
                                  == jz]['filtEff'].values[0]
        cross_section = jz_description[jz_description['JZ']
                                       == jz]['crossSection [pb]'].values[0]
        norm_name = jz_description[jz_description['JZ']
                                   == jz]['Norm'].values[0]

        cross_section = tf.constant(cross_section)
        filt_eff = tf.constant(filt_eff)
        lumi = tf.constant(lumi)
        jz_df = load_dataframe(
            file, variables, cross_section, filt_eff, lumi, norm_name)
        jz_df['JZ_slice'] = jz
        df.append(jz_df)
    df = pd.concat(df)
    return df


def main(args: argparse.Namespace) -> None:
    variables = ['pt', 'eta', 'phi', 'e', 'm']
    try:
        print('Loading from csv...')
        df = pd.read_csv(os.path.join(
            args.load_path, f"pythia_physical_{args.phys_object}.csv"))
    except FileNotFoundError:
        print('Saved csv not found, creating new one...')
        df = load_all_jzs(args.load_path, args.dataset_type, variables,
                          args.jz_description, args.phys_object)
        df.columns = [col.replace(f'{args.phys_object}_', '')
                      for col in df.columns]
        print('Saving csv...')
        df.to_csv(os.path.join(args.load_path,
                  f"pythia_physical_{args.phys_object}.csv"))

    for var in variables:
        print(f'Plotting {var}...')
        sns_plot_phys(df, os.path.join(args.save_path,
                      args.dataset_type), var, args.phys_object)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = args.load_path
    main(args)
    print('Done!')
