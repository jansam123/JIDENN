

MODEL_NAMING_SCHEMA = {
    'basic_fc': 'Fully Connected',
    'depart': 'DeParT',
    'part': 'ParT',
    'highway': 'Highway',
    'depart_rel': 'DeParT (Relative)',
    'interacting_depart': 'Interacting DeParT',
    'interacting_depart_no_norm': 'Interacting DeParT (No Norm)',
    'interacting_part': 'Interacting ParT',
    'pfn': 'PFN',
    'efn': 'EFN',
    'transformer': 'Transformer',
}


MATRIC_NAMING_SCHEMA = {
    'binary_accuracy': 'Accuracy',
    'auc': 'AUC',
    'gluon_efficiency': r'$\varepsilon_g$',
    'quark_efficiency': r'$\varepsilon_q$',
    'gluon_rejection': r'$\varphi_g$',
    'quark_rejection': r'$\varphi_q$',
    'gluon_rej_at_quark_eff_0.9': r'$\varphi_g @_{0.9} \varepsilon_q$',
    'quark_rej_at_gluon_eff_0.9': r'$\varphi_q @_{0.9} \varepsilon_g$',
    'loss': 'Loss',
}
