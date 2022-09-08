from .get_dataset import get_gluon_dataset, get_quark_dataset
from .DataSchema import DataSchema
from .preprocess import pipe

def get_qg_dataset(files: list[str],
                   batch_size:int, 
                   cut: str | None,
                   take:int | None, 
                   shuffle_buffer:int | None):
    
    data_schema = DataSchema()
    gluon_dataset = get_gluon_dataset(data_schema, files, cut).dataset
    quark_dataset = get_quark_dataset(data_schema, files, cut).dataset
    
    return pipe([gluon_dataset, quark_dataset], [0.5, 0.5], batch_size, take, shuffle_buffer, label_mapping=data_schema.label_mapping)