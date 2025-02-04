import re
import os
import sys
sys.path.append(os.getcwd())
import argparse
import logging
from functools import partial
from typing import Callable,Optional 
import tensorflow as tf
import pandas as pd
#
from jidenn.preprocess.flatten_dataset import flatten_dataset
from jidenn.preprocess.resampling import write_new_variable
from jidenn.preprocess.resampling import get_cut_fn
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables

logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, nargs='+',
                    help="Path to load the dataset")
parser.add_argument("--shuffle", type=int, default=100_000,
                    required=False, help="Shuffle buffer size")
parser.add_argument("--sample_name_index", type=int, default=-1,)
parser.add_argument("--weight_upper_cut", type=float, 
                    help="Weight upper cut")
parser.add_argument("--weight_lower_cut", type=float, 
                    help="Weight lower cut")
parser.add_argument("--take", type=int, default=1_000_000,
                    help="Number of jets to take")
parser.add_argument("--num_shards", type=int, default=256, required=False,
                    help="Number of shards to save the dataset in")
parser.add_argument("--jz_slicing", action='store_true',help="Use JZ slicing")
parser.add_argument("--max_idx", type=int, default=None,
                    help="Maximum index of jets to use, select 2 to use leading and subleading")
parser.add_argument("--eta_cut", type=float, default=2.1, help="Eta cut")
parser.add_argument("--pt_lower_cut", type=float,
                    default=0.2e6, help="Pt lower cut")
parser.add_argument("--pt_upper_cut", type=float,
                    default=2.5e6, help="Pt upper cut")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--sample_description_file", type=str,
                    default='data/sample_description.csv', help="JZ description csv file.")
parser.add_argument("--not_stop_on_empty_dataset", action='store_true',
                    help="Do not stop if a dataset is empty")


JZ_LOW_PT = [20, 60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300]
JZ_LOW_PT = [val * 1e3 for val in JZ_LOW_PT]

JZ_HIGH_PT = [60, 160, 400, 800, 1300, 1800,
              2500, 3200, 3900, 4600, 5300, 7000]
JZ_HIGH_PT = [val * 1e3 for val in JZ_HIGH_PT]

MAX_FRAC = 1.4


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

def get_process_subsample(per_subsample_filter: Optional[Callable[[int], Callable[[ROOTVariables], bool]]] = None, subsample_id_variable_name: Optional[str] = 'subsample_id'):
    @tf.function
    def _process_subsample(dataset: tf.data.Dataset, weight_info) -> tf.data.Dataset:
            cross_section, filt_eff, lumi, norm, sub_sample = weight_info
            dataset = dataset.filter(per_subsample_filter(
                sub_sample)) if per_subsample_filter is not None else dataset
            dataset = dataset.map(lambda x: {var: x[var] for var in x.keys() if 'HLT' not in var})
            dataset = dataset.map(lambda x: {var: x[var] for var in x.keys() if var not in [
                                  'jets_e', 'jets_Constituent_e', 'jets_TopoTower_e', 'muons_e', 'photons_e']})
            # dataset = dataset.map(lambda x: {
            #         **x, 'jets_e': tf.sqrt(x['jets_pt']**2 + x['jets_m']**2 + tf.sinh(x['jets_eta'])**2), 'jets_Constituent_e': tf.sqrt(x['jets_Constituent_pt']**2 + x['jets_Constituent_m']**2 + tf.sinh(x['jets_Constituent_eta'])**2),
            #         'jets_TopoTower_e': tf.sqrt(x['jets_TopoTower_pt']**2 + x['jets_TopoTower_m']**2 + tf.sinh(x['jets_TopoTower_eta'])**2),})
            dataset = dataset.map(write_weights(cross_section, filt_eff, lumi, norm))
            if subsample_id_variable_name is not None:
                dataset = dataset.map(write_new_variable(variable_name=subsample_id_variable_name,variable_value=sub_sample))
            return dataset
    return _process_subsample


def get_jz_number(description):
    match = re.search('JZ(\d+)', description)
    if match:
        return int(match.group(1))
    else:
        return None
    
def main(args: argparse.Namespace) -> None:
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

    sample_description_file = pd.read_csv(args.sample_description_file)
    logging.info(sample_description_file)
    lumi = 139_000
    
    if isinstance(args.load_path, str):
        args.load_path = [args.load_path]

    sizes = []
    subsample_info = []
    for subsample_id, subsample in enumerate(args.load_path):
        subsample_name = subsample.split('/')[args.sample_name_index]
        if subsample_name not in sample_description_file['Description'].values:
            raise ValueError(f"Subsample {subsample_name} not found in sample description file") 
        
        if args.jz_slicing:
            subsample_id = tf.constant(get_jz_number(subsample_name), dtype=tf.int32)
            logging.info(subsample_id)
        filt_eff = tf.constant(sample_description_file[sample_description_file['Description'] == subsample_name]['filtEff'].values[0])
        cross_section = tf.constant(sample_description_file[sample_description_file['Description'] == subsample_name]['crossSection [pb]'].values[0])
        norm_name = sample_description_file[sample_description_file['Description'] == subsample_name]['Norm'].values[0]

        dataset = JIDENNDataset.load(subsample)
        size = dataset.length
        norm = tf.constant(dataset.metadata[norm_name])

        logging.info(f'Cross section: {cross_section}, Filter efficiency: {filt_eff}, Luminosity: {lumi}, Norm: {norm}, Subsample id: {subsample_id}')
        subsample_info.append((cross_section, filt_eff, lumi, norm, subsample_id))
        sizes.append(size)

    # sizes = [217, 218, 333, 409, 183, 30, 15]
    # sizes = [215, 216, 331, 407, 182, 30, 15]
    logging.info(sizes)
    sampling_weights = [size / sum(sizes) for size in sizes]
    logging.info(sampling_weights)
    
    if args.jz_slicing:
        per_subsample_filter = lambda jz: get_cut_fn('jets_pt', upper_limit=MAX_FRAC * tf.constant(JZ_HIGH_PT)[jz - 1])
        subsample_id_variable_name = 'JZ_slice'
    else:
        per_subsample_filter = None
        subsample_id_variable_name = None
        
    if len(args.load_path) > 1:
        dataset = JIDENNDataset.load_multiple(args.load_path, dataset_mapper=get_process_subsample(per_subsample_filter=per_subsample_filter, subsample_id_variable_name=subsample_id_variable_name),
                                            file_labels=subsample_info, weights=sampling_weights, mode='sample', rerandomize_each_iteration=False, stop_on_empty_dataset=not args.not_stop_on_empty_dataset)
    else:
        dataset = JIDENNDataset.load(args.load_path[0])
        dataset = dataset.remap_data(write_weights(cross_section, filt_eff, lumi, norm))
    
    if args.weight_upper_cut is not None or args.weight_lower_cut is not None:
        dataset = dataset.filter(get_cut_fn('weight', lower_limit=args.weight_lower_cut, upper_limit=args.weight_upper_cut))
    
    @tf.function
    def is_central_jet(data):
        # get index of the central jet, considering only the first two jets
        data = data.copy() 
        central_jet_idx = tf.argmin(tf.abs(data['jets_eta'][:2]), axis=0)
        data['jets_isCentral'] = tf.one_hot(central_jet_idx, tf.shape(data['jets_eta'])[0], dtype=tf.int32)
        return data
         
    dataset = dataset.remap_data(is_central_jet)
    
    dataset = dataset.apply(partial(flatten_dataset, reference_variable=args.reference_variable, max_idx=args.max_idx,
                                    wanted_values=args.wanted_values, variables=['jets_eta', 'jets_pt'], lower_cuts=[-args.eta_cut, args.pt_lower_cut], upper_cuts=[args.eta_cut, args.pt_upper_cut]))
    dataset = dataset.take(args.take)

    dataset = dataset.apply(lambda x: x.shuffle(
        args.shuffle).prefetch(tf.data.AUTOTUNE))
    os.makedirs(args.save_path, exist_ok=True)
    dataset.save(args.save_path, num_shards=args.num_shards)
    logging.info(f'Saved dataset to {args.save_path}')
    dataset = JIDENNDataset.load(args.save_path)
    size = dataset.length
    logging.info(f"Number of jets: {size}")
    dataset.plot_single_variable('jets_pt',
                                 weight_variable='weight',
                                 save_path=os.path.join(
                                     args.save_path, 'pt_noJZ.png'),
                                 ylog=True,
                                 badge_text='$N_{\mathrm{jets}}$ = ' +
                                 f'{size:,} \n',
                                 bins=100,)
    dataset.plot_single_variable('jets_pt',
                                 weight_variable='weight',
                                 save_path=os.path.join(
                                     args.save_path, 'pt_label.png'),
                                 ylog=True,
                                 badge_text='$N_{\mathrm{jets}}$ = ' +
                                 f'{size:,} \n',
                                 bins=100,
                                 multiple='layer',
                                 hue_variable='jets_PartonTruthLabelID')
    dataset.plot_single_variable('weight',
                                save_path=os.path.join(
                                    args.save_path, 'weight.png'),
                                ylog=True,
                                bins=100,
                                multiple='stack')
    if args.jz_slicing and len(args.load_path) > 1:
        dataset.plot_single_variable('jets_pt',
                                    weight_variable='weight',
                                    save_path=os.path.join(
                                        args.save_path, 'pt.png'),
                                    ylog=True,
                                    badge_text='$N_{\mathrm{jets}}$ = ' +
                                    f'{size:,} \n',
                                    bins=100,
                                    multiple='stack',
                                    hue_variable='JZ_slice')
        dataset.plot_single_variable('jets_pt',
                                    weight_variable='weight',
                                    save_path=os.path.join(
                                        args.save_path, 'pt_not_comm_norm.png'),
                                    ylog=True,
                                    badge_text='$N_{\mathrm{jets}}$ = ' +
                                    f'{size:,} \n',
                                    bins=100,
                                    multiple='stack',
                                    common_norm=False,
                                    hue_variable='JZ_slice')
        dataset.apply(lambda x: x.filter(lambda y: y['jets_PartonTruthLabelID'] != 21)).plot_single_variable('jets_pt',
                                                                                                            weight_variable='weight',
                                                                                                            save_path=os.path.join(
                                                                                                                args.save_path, 'pt_quark.png'),
                                                                                                            ylog=True,
                                                                                                            badge_text='quark\n',
                                                                                                            bins=100,
                                                                                                            multiple='stack',
                                                                                                            hue_variable='JZ_slice')
        dataset.apply(lambda x: x.filter(lambda z: z['jets_PartonTruthLabelID'] == 21)).plot_single_variable('jets_pt',
                                                                                                            weight_variable='weight',
                                                                                                            save_path=os.path.join(
                                                                                                                args.save_path, 'pt_gluon.png'),
                                                                                                            ylog=True,
                                                                                                            badge_text='gluon\n',
                                                                                                            bins=100,
                                                                                                            multiple='stack',
                                                                                                            hue_variable='JZ_slice')
        dataset.plot_single_variable('jets_pt',
                                    weight_variable=None,
                                    save_path=os.path.join(
                                        args.save_path, 'pt_noW.png'),
                                    ylog=True,
                                    badge_text='$N_{\mathrm{jets}}$ = ' +
                                    f'{size:,} \n',
                                    bins=100,
                                    multiple='stack',
                                    hue_variable='JZ_slice')
        dataset.plot_single_variable('weight',
                                    save_path=os.path.join(
                                        args.save_path, 'weight_JZ.png'),
                                    ylog=True,
                                    bins=100,
                                    multiple='stack',
                                    hue_variable='JZ_slice')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

 