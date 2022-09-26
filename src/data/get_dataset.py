from .JIDENNDataset import JIDENNDataset
from src.config import config_subclasses as cfg        
        
def get_gluon_dataset(args_data:cfg.Data, 
                      files:list[str],
                      ) -> JIDENNDataset:
    
    cut = f"({args_data.cut})" if args_data.cut is not None else ''
    cut += f"&({args_data.target}=={args_data.raw_gluon})"
    cut = cut[1:] if cut[0] == '&' else cut
    
    return JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=1,
                         cut=cut,)
                         
                         
def get_quark_dataset(args_data:cfg.Data, 
                      files:list[str],
                      ) -> JIDENNDataset:
    
    cut = f"({args_data.cut})" if args_data.cut is not None else ''
    cut += f'&({"".join([f"({args_data.target}=={q})|" for q in args_data.raw_quarks])[:-1]})'
    
    cut = cut[1:] if cut[0] == '&' or cut[0] == '|' else cut

    return JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=1,
                         cut=cut,)

def get_mixed_dataset(args_data:cfg.Data, 
                      files:list[str],
                      ) -> JIDENNDataset:
    
    cut = f"({args_data.cut})" if args_data.cut is not None else ''
    cut += f"&({args_data.target}=={args_data.raw_gluon})"
    
    cut = f'({"".join([f"({args_data.target}=={q})|" for q in args_data.raw_quarks])[:-1]})'
    cut = cut[1:] if cut[0] == '&' else cut
        
    
    return JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=1,
                         cut=cut,)
                         
                         
                         
