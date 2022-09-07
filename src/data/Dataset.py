from .get_dataset import get_gluon_dataset, get_quark_dataset
from .DataSchema import DataSchema
from .preprocess import pipe
from typing import Optional, List

def get_qg_dataset(files: List[str],
                   batch_size:int, 
                   cut: Optional[str],
                   take:Optional[int], 
                   shuffle_buffer:Optional[int]):
    
    data_schema = DataSchema()
    gluon_dataset = get_gluon_dataset(data_schema, files, cut).dataset
    quark_dataset = get_quark_dataset(data_schema, files, cut).dataset
    
    return pipe([gluon_dataset, quark_dataset], [0.5, 0.5], batch_size, take, shuffle_buffer, label_mapping=data_schema.label_mapping)