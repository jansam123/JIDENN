import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import argparse
import logging
import numpy as np
from typing import Optional, Callable
from functools import partial
#
#
from jidenn.preprocess.resampling import resample_var_with_labels, write_new_variable, get_cut_fn, get_filter_fn, resample_labels, resample_2d_var, resample_2d_var_with_labels
from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.preprocess.flatten_dataset import flatten_dataset

# from jidenn.evaluation.plotter import plot_single_dist


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, nargs='+',
                    help="Path to the root file")
parser.add_argument("--load_path_labels", type=str, nargs='*',
                    help="Labels used for multiple load_paths.")
parser.add_argument("--sampling_weights", type=float, nargs='*', default=None,
                    help="Weights used for multiple load_paths.")
parser.add_argument("--subsample_id_variable_name", type=str, default='subsample_id',
                    help="Name of the variable to use as subsample id")
parser.add_argument("--load_path_uniform_sampling", action='store_true',
                    help="Use uniform sampling for multiple load_paths")
parser.add_argument("--not_stop_on_empty_dataset", action='store_true',
                    help="Do not stop if a dataset is empty")
parser.add_argument("--num_shards", type=int, default=256,
                    help="Number of shards to use when saving the dataset")
parser.add_argument("--take", type=int, default=None,
                    help="Number of samples to take")
parser.add_argument("--shuffle", type=int, default=1_000,
                    required=False, help="Shuffle buffer size")
# parser.add_argument("--max_jz", type=int, default=10, help="Maximum JZ to use")
# parser.add_argument("--min_jz", type=int, default=2, help="Maximum JZ to use")
#
parser.add_argument("--bins", type=int, default=100, nargs='+',
                    help="Number of bins to use for the binning")
parser.add_argument("--flattening_var", type=str, default='jets_pt',  nargs='+',
                    help="Variable to flatten.")
parser.add_argument('-w', "--weight_var", type=str, default=None,
                    help="Use reweighting, specify the weight variable")
parser.add_argument("--min_count", action='store_true',
                    help="Use min count for each bin")
#
parser.add_argument("--flattening_reference_variable", type=str,
                    help="Variable to use as reference for flattening")
parser.add_argument("--flat_var_upper_limit", type=float,
                    help="Upper limit for the variable to flatten")
parser.add_argument("--flat_var_lower_limit", type=float,
                    help="Lower limit for the variable to flatten")
#

parser.add_argument("--eta_cut", type=float, default=2.1, help="Eta cut")
parser.add_argument("--pt_lower_cut", type=float,
                    default=0.2e6, help="Pt lower cut")
parser.add_argument("--pt_upper_cut", type=float,
                    default=2.5e6, help="Pt upper cut")
parser.add_argument("--jz_slicing", action='store_true', help="Use JZ slicing")
parser.add_argument("--log_binning", action='store_true',
                    help="Use log-spaced bins")
parser.add_argument("--precompute", action='store_true',
                    help="Precompute the initial distribution")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--max_idx", type=int, default=2,
                    help="Maximum index of jets to use, select 2 to use leading and subleading")

JZ_LOW_PT = [20, 60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300]
JZ_LOW_PT = [val * 1e3 for val in JZ_LOW_PT]

JZ_HIGH_PT = [60, 160, 400, 800, 1300, 1800,
              2500, 3200, 3900, 4600, 5300, 7000]
JZ_HIGH_PT = [val * 1e3 for val in JZ_HIGH_PT]


@tf.function
def get_labeled_pt(data):
    parton = data['jets_PartonTruthLabelID']
    if tf.equal(parton, tf.constant(21)):
        label = 0
    elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
        label = 1
    else:
        label = -1
    return {'jets_pt': data['jets_pt'], 'label': label, 'all_label': parton}

@tf.function
def is_central_jet(data):
    # get index of the central jet, considering only the first two jets
    data = data.copy() 
    central_jet_idx = tf.argmin(tf.abs(data['jets_eta'][:2]), axis=0)
    data['jets_isCentral'] = tf.one_hot(central_jet_idx, tf.shape(data['jets_eta'])[0], dtype=tf.int32)
    return data


def get_process_subsample(subsample_id_variable_name: Optional[str] = 'subsample_id',
                          reference_variable: Optional[str] = 'jets_PartonTruthLabelID',
                          wanted_values: Optional[list] = [
                              1, 2, 3, 4, 5, 6, 21],
                          max_idx: Optional[int] = 2,
                          cut_variables: Optional[list] = [
                              'jets_pt', 'jets_eta'],
                          lower_cuts: Optional[list] = [20_000, -4.5],
                          upper_cuts: Optional[list] = [2.5e6, 4.5],
                          filter_fn: Optional[Callable] = None
                          ):
    
    def _process_subsample(dataset: tf.data.Dataset, id) -> tf.data.Dataset:
        if subsample_id_variable_name is not None and id is not None:
            dataset = dataset.map(write_new_variable(
                variable_name=subsample_id_variable_name, variable_value=id))
        dataset = dataset.map(
            lambda x: {var: x[var] for var in x.keys() if 'HLT' not in var})
        dataset = dataset.map(lambda x: {var: x[var] for var in x.keys() if var not in [
                                'jets_e', 'jets_Constituent_e', 'jets_TopoTower_e', 'muons_e', 'photons_e', 'jets_JvtSFEff', 
                                'jets_JvtSFIneff', 'jets_FJvtSFIneff', 'jets_FJvtSFEff',"jets_passFJvt", "jets_passJvt",]})
        dataset = dataset.map(is_central_jet)
        dataset = dataset.filter(lambda x: tf.shape(x['jets_pt'])[0] > 0)
         
        dataset = dataset.apply(partial(flatten_dataset, reference_variable=reference_variable, max_idx=max_idx,
                                wanted_values=wanted_values, variables=cut_variables, lower_cuts=lower_cuts, upper_cuts=upper_cuts))
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn(id))
        return dataset
    return _process_subsample


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(
        f'Running with args: {{{", ".join([f"{k}: {v}" for k, v in vars(args).items()])}}}')

    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()
    if isinstance(args.flattening_var, (list, tuple)) and len(args.flattening_var) == 1:
        args.flattening_var = args.flattening_var[0]
    if isinstance(args.flattening_var, (list, tuple)) and len(args.flattening_var) > 1:
        if len(args.flattening_var) != len(args.bins):
            raise ValueError(
                'If flattening_var is a list, it must have the same length as bins')
        if len(args.flattening_var) > 2:
            raise ValueError(
                'Only one or two flattening variables are supported')
        
        if args.flattening_var[0] == 'jets_pt':
            args.flattening_reference_variable = 'jets_PartonTruthLabelID'
            flat_var_upper_limit = args.pt_upper_cut
            flat_var_lower_limit = args.pt_lower_cut
        elif args.flattening_var[0] == 'jets_eta':
            args.flattening_reference_variable = 'jets_PartonTruthLabelID'
            flat_var_upper_limit = args.eta_cut
            flat_var_lower_limit = -args.eta_cut
        
        if args.flattening_var[1] == 'jets_pt':
            flat_var_upper_limit = (flat_var_upper_limit, args.pt_upper_cut)
            flat_var_lower_limit = (flat_var_lower_limit, args.pt_lower_cut)
        elif args.flattening_var[1] == 'jets_eta':
            flat_var_upper_limit = (flat_var_upper_limit, args.eta_cut)
            flat_var_lower_limit = (flat_var_lower_limit, -args.eta_cut)
        
        
    elif args.flattening_var == 'jets_pt':
        args.flattening_reference_variable = 'jets_PartonTruthLabelID'
        flat_var_upper_limit = args.pt_upper_cut
        flat_var_lower_limit = args.pt_lower_cut

    if args.jz_slicing:
        file_labels = [int(label) for label in args.load_path_labels]
        # filter_fn = lambda jz: get_cut_fn(
        #     'jets_pt', lower_limit=JZ_LOW_PT[jz - 1], upper_limit=JZ_HIGH_PT[jz - 1])
        filter_fn = None
        subsample_id_variable_name = 'JZ_slice'
    elif args.load_path_labels is not None and isinstance(args.load_path_labels, (list, tuple)):
        filter_fn = None
        file_labels = args.load_path_labels
        subsample_id_variable_name = args.subsample_id_variable_name
    elif isinstance(args.load_path, (list, tuple)) and len(args.load_path) > 1:
        filter_fn = None
        file_labels = [f'file_{i}' for i in range(len(args.load_path))]
        subsample_id_variable_name = args.subsample_id_variable_name
    else:
        filter_fn = None
        file_labels = None
        subsample_id_variable_name = args.subsample_id_variable_name
    dataset_mapper = get_process_subsample(subsample_id_variable_name=subsample_id_variable_name,
                                                                        reference_variable=args.reference_variable,
                                                                        wanted_values=args.wanted_values,
                                                                        max_idx=args.max_idx if args.max_idx > 0 else None,
                                                                        cut_variables=[
                                                                            'jets_pt', 'jets_eta'],
                                                                        lower_cuts=[
                                                                            args.pt_lower_cut, -abs(args.eta_cut)],
                                                                        upper_cuts=[
                                                                            args.pt_upper_cut, abs(args.eta_cut)],
                                                                        filter_fn=filter_fn)
    
    if isinstance(args.load_path, (list, tuple)) and len(args.load_path) > 1:
        if len(args.load_path) != len(args.load_path):
            raise ValueError(
                'If load_path_labels is a list, it must have the same length as load_path')

        if args.load_path_uniform_sampling:
            sizes = [1 for _ in args.load_path]
        else:
            sizes = [JIDENNDataset.load(subsample).length for subsample in args.load_path]
        logging.info(f'Dataset sizes: {sizes}')
        if args.sampling_weights is None:
            sampling_weights = [size / sum(sizes) for size in sizes]
        else:
            sampling_weights = args.sampling_weights
        logging.info(sampling_weights)


        dataset = JIDENNDataset.load_multiple(args.load_path, file_labels=file_labels, stop_on_empty_dataset=not args.not_stop_on_empty_dataset, mode='sample', rerandomize_each_iteration=False,dataset_mapper=dataset_mapper, weights=sampling_weights)
    else:
        dataset = JIDENNDataset.load(args.load_path)
        dataset = dataset.apply(lambda x: dataset_mapper(x, None))
        

    dataset = dataset.take(args.take) if args.take is not None and args.take > 0 else dataset

        

    if isinstance(args.flattening_var, (list, tuple)) and len(args.flattening_var) > 1:
        resampler = partial(resample_2d_var_with_labels,
                            bins1=args.bins[0],
                            lower_var_limit1=flat_var_lower_limit[0],
                            upper_var_limit1=flat_var_upper_limit[0],
                            variable1=args.flattening_var[0],
                            bins2=args.bins[1],
                            lower_var_limit2=flat_var_lower_limit[1],
                            upper_var_limit2=flat_var_upper_limit[1],
                            variable2=args.flattening_var[1],
                            label_variable=args.flattening_reference_variable,
                            precompute_init_dist=args.precompute,
                            weight_var=args.weight_var,
                            from_min_count=args.min_count)
    else:
        resampler = partial(resample_var_with_labels,
                            bins=args.bins if isinstance(
                                args.bins, int) else args.bins[0],
                            lower_var_limit=flat_var_lower_limit if isinstance(
                                flat_var_lower_limit, (int, float)) else flat_var_lower_limit[0],
                            upper_var_limit=flat_var_upper_limit if isinstance(
                                flat_var_upper_limit, (int, float)) else flat_var_upper_limit[0],
                            log_binning_base=np.e if args.log_binning else None,
                            variable=args.flattening_var if isinstance(
                                args.flattening_var, str) else args.flattening_var[0],
                            label_variable=args.flattening_reference_variable,
                            precompute_init_dist=args.precompute,
                            weight_var=args.weight_var,
                            from_min_count=args.min_count)
        
    dataset = dataset.apply(resampler)

    dataset = dataset.apply(lambda x: x.shuffle(
        args.shuffle, seed=42).prefetch(tf.data.AUTOTUNE))
    os.makedirs(args.save_path, exist_ok=True)
    dataset.save(args.save_path, num_shards=args.num_shards)

    dataset = JIDENNDataset.load(args.save_path)
    print(f'Dataset size: {dataset.length}')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

