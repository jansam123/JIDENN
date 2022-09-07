from .JIDENNDataset import JIDENNDataset
from .DataSchema import DataSchema      
from typing import Optional, List  
        
def get_gluon_dataset(schema:DataSchema, 
                      files:List[str], 
                      cut:Optional[str]=None) -> JIDENNDataset:
    
    cut = f"({cut})" if cut is not None else ''
    cut += f"&({schema.target}=={schema.raw_qluon})"
    cut = cut[1:] if cut[0] == '&' else cut
    
    return JIDENNDataset(files=files,
                         variables=schema.variables,
                         target=schema.target,
                         weight=schema.weight,
                         reading_size=1000,
                         num_workers=1,
                         cut=cut,)
                         
                         
def get_quark_dataset(schema:DataSchema, 
                      files:List[str], 
                      cut:Optional[str]=None) -> JIDENNDataset:
    
    cut = f"({cut})" if cut is not None else ''
    cut += f'&({"".join([f"({schema.target}=={q})|" for q in schema.raw_quarks])[:-1]})'
    
    cut = cut[1:] if cut[0] == '&' or cut[0] == '|' else cut

    return JIDENNDataset(files=files,
                         variables=schema.variables,
                         target=schema.target,
                         weight=schema.weight,
                         reading_size=1000,
                         num_workers=1,
                         cut=cut,)

def get_mixed_dataset(schema:DataSchema, 
                      files:List[str], 
                      cut:Optional[str]=None) -> JIDENNDataset:
    
    cut = f"({cut})" if cut is not None else ''
    cut += f"&({schema.target}=={schema.raw_qluon})"
    
    cut = f'({"".join([f"({schema.target}=={q})|" for q in schema.raw_quarks])[:-1]})'
    cut = cut[1:] if cut[0] == '&' else cut
        
    
    return JIDENNDataset(files=files,
                         variables=schema.variables,
                         target=schema.target,
                         weight=schema.weight,
                         reading_size=1000,
                         num_workers=1,
                         cut=cut,)
                         
                         
                         
