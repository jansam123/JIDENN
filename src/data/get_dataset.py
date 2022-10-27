from typing import Callable
from .JIDENNDataset import JIDENNDataset
from src.config import config_subclasses as cfg
from src.data.preprocess import pipe
import tensorflow as tf
from src.data.utils.Cut import Cut


def get_dataset(files: list[str],
                args_data: cfg.Data,
                filter: Callable | None = None) -> tf.data.Dataset:

    dataset = JIDENNDataset(files=files,
                            variables=args_data.variables,
                            target=args_data.target,
                            weight=args_data.weight,
                            reading_size=args_data.reading_size,
                            num_workers=args_data.num_workers,
                            cut=args_data.cut,
                            filter=filter).dataset

    # dataset = dataset.filter(filter) if filter is not None else dataset
    return dataset


def get_preprocessed_dataset(files: list[list[str]],
                             args_data: cfg.Data,
                             name: str,
                             args_dataset: cfg.Dataset | None = None,
                             size: int | None = None,
                             additial_cut: str | None = None) -> tf.data.Dataset:

    def to_dict(x, y, z):
        sample_dict = {var: x[i] for i, var in enumerate(args_data.variables.perJet)}
        sample_dict.update({var: x[i] for i, var in enumerate(args_data.variables.perEvent)})
        sample_dict.update({args_data.target: y})
        if args_data.weight is not None:
            sample_dict.update({args_data.weight: z})
        return sample_dict

    gluon_cut = Cut(f'{args_data.target}=={args_data.raw_gluon}')
    if args_data.cut is not None:
        gluon_cut &= Cut(args_data.cut)
    if additial_cut is not None:
        gluon_cut &= Cut(additial_cut)

    quark_cut = Cut(' || '.join([f'({args_data.target}=={var})' for var in args_data.raw_quarks]))
    if args_data.cut is not None:
        quark_cut &= Cut(args_data.cut)
    if additial_cut is not None:
        quark_cut &= Cut(additial_cut)

    mixed_cut = Cut(f'{args_data.target}=={args_data.raw_gluon}') | Cut(
        ' || '.join([f'({args_data.target}=={var})' for var in args_data.raw_quarks]))
    if args_data.cut is not None:
        mixed_cut &= Cut(args_data.cut)
    if additial_cut is not None:
        mixed_cut &= Cut(additial_cut)

    def gen_single_JZ(files, q_filter, g_filter):
        gluon_dataset = get_dataset(files, args_data, filter=g_filter)
        quark_dataset = get_dataset(files, args_data, filter=q_filter)
        return tf.data.Dataset.sample_from_datasets([gluon_dataset, quark_dataset], [0.5, 0.5], stop_on_empty_dataset=True)

    if args_data.JZ_cut is not None:
        q_JZ_cuts = [(quark_cut & Cut(cut)).get_filter_function(to_dict) for cut in args_data.JZ_cut]
        g_JZ_cuts = [(gluon_cut & Cut(cut)).get_filter_function(to_dict) for cut in args_data.JZ_cut]
    else:
        q_JZ_cuts = [quark_cut.get_filter_function(to_dict)]*len(files)
        g_JZ_cuts = [gluon_cut.get_filter_function(to_dict)]*len(files)
    datasets = [gen_single_JZ(subfiles, q_cut, g_cut) for subfiles, q_cut, g_cut in zip(files, q_JZ_cuts, g_JZ_cuts)]
    
    if len(datasets) > 1:
        dataset = tf.data.Dataset.sample_from_datasets(datasets, stop_on_empty_dataset=True, weights=args_data.JZ_weights)
    else:
        dataset = datasets[0]

    @tf.function
    def label_mapping(x):
        if x == args_data.raw_gluon:
            return args_data.gluon
        else:
            return args_data.quark
        
    dataset = dataset.map(lambda x, y, z: (x, label_mapping(y), z), num_parallel_calls=tf.data.AUTOTUNE)
    
    if args_dataset is not None:
        return dataset.apply(pipe(args_dataset=args_dataset, name=name, take=size))
    elif size is not None:
        return dataset.take(size)
    else:
        return dataset
