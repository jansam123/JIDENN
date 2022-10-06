from .get_dataset import get_gluon_dataset, get_quark_dataset
from src.config import config_subclasses as cfg        
from .preprocess import pipe

def get_qg_dataset(files: list[str],
                   args_data: cfg.Data,
                   args_dataset: cfg.Dataset,
                   name:str,
                   ):
    
    gluon_dataset = get_gluon_dataset(args_data, files)
    quark_dataset = get_quark_dataset(args_data, files)

    
    def label_mapping(x):
        if x == args_data.raw_gluon:
            return args_data.gluon
        else:
            return args_data.quark

    return pipe([gluon_dataset, quark_dataset], [0.5, 0.5], args_dataset, label_mapping=label_mapping, name=name)