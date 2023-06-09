LATEX_NAMING_CONVENTION = {
    'jets_ActiveArea4vec_m': 'jets_ActiveArea4vec_m',
    'jets_ActiveArea4vec_phi': 'jets_ActiveArea4vec_phi',
    'jets_ActiveArea4vec_eta': 'jets_ActiveArea4vec_eta',
    'jets_ActiveArea4vec_pt': 'jets_ActiveArea4vec_pt',
    'jets_DetectorEta': 'jets_DetectorEta',
    'jets_FracSamplingMax': 'jets_FracSamplingMax',
    'jets_FracSamplingMaxIndex': 'jets_FracSamplingMaxIndex',
    'jets_GhostMuonSegmentCount': 'jets_GhostMuonSegmentCount',
    'jets_JVFCorr': 'jets_JVFCorr',
    'jets_JetConstitScaleMomentum_eta': 'jets_JetConstitScaleMomentum_eta',
    'jets_JetConstitScaleMomentum_m': 'jets_JetConstitScaleMomentum_m',
    'jets_JetConstitScaleMomentum_phi': 'jets_JetConstitScaleMomentum_phi',
    'jets_JetConstitScaleMomentum_pt': 'jets_JetConstitScaleMomentum_pt',
    'jets_JvtRpt': 'jets_JvtRpt',
    'jets_fJVT': 'jets_fJVT',
    'jets_passFJVT': 'jets_passFJVT',
    'jets_passJVT': 'jets_passJVT',
    'jets_Timing': 'jets_Timing',
    'jets_Jvt': 'jets_Jvt',
    'jets_EMFrac': 'jets_EMFrac',
    'jets_Width': 'jets_Width',
    'jets_chf': 'jets_chf',
    'jets_eta': 'jets_eta',
    'jets_m': 'jets_m',
    'jets_phi': 'jets_phi',
    'jets_pt': 'jets_pt',
    'jets_PFO_n': 'jets_PFO_n',
    'jets_ChargedPFOWidthPt1000[0]': 'jets_ChargedPFOWidthPt1000',
    'jets_TrackWidthPt1000[0]': 'jets_TrackWidthPt1000',
    'jets_NumChargedPFOPt1000[0]': 'jets_NumChargedPFOPt1000',
    'jets_NumChargedPFOPt500[0]': 'jets_NumChargedPFOPt500',
    'jets_SumPtChargedPFOPt500[0]': 'jets_SumPtChargedPFOPt500',
    'jets_PFO_m': 'jets_PFO_m',
    'jets_PFO_pt': 'jets_PFO_pt',
    'jets_PFO_eta': 'jets_PFO_eta',
    'jets_PFO_phi': 'jets_PFO_phi',
    'corrected_averageInteractionsPerCrossing[0]': 'corrected_averageInteractionsPerCrossing',
    'pt_jet': 'pt_jet',
    'eta_jet': 'eta_jet',
    'N_PFO': 'N_PFO',
    'W_PFO_jet': 'W_PFO_jet',
    'C1_PFO_jet': 'C1_PFO_jet',
    'log_pT': r'$\log{p_{\mathrm{T}}}$',
    'log_PT|PTjet': r'$\log{\frac{p_{\mathrm{T}}}{p_{\mathrm{T}}^{\mathrm{jet}}}}$',
    'log_E': r'$\log{E}$',
    'log_E|Ejet': r'$\log{\frac{E}{E^{\mathrm{jet}}}}$',
    'm': r'$m$',
    'deltaEta': r'$\Delta \eta$',
    'deltaPhi': r'$\Delta \phi$',
    'deltaR': r'$\Delta R$',
    'delta': r'$\log{\Delta}$',
    'k_t': r'$\log{k_{\mathrm{T}}}$',
    'z': r'$z$',
    'm2': r'$\log{m^2}$',
    'log_norm_p': r'$\log{\norm{p}}$',
}


MODEL_NAMING_SCHEMA = {
    'depart_flat_pT': r'DeParT (Flat $p_{\mathrm{T}}$)',
    'fc': 'Fully Connected',
    'basic_fc': 'Fully Connected',
    'depart': 'DeParT',
    'part': 'ParT',
    'highway': 'Highway',
    'depart_rel': 'DeParT (Relative)',
    'interacting_depart': 'iDeParT',
    'interacting_depart_no_norm': 'iDeParT (No Norm)',
    'interacting_part': 'iParT',
    'pfn': 'PFN',
    'efn': 'EFN',
    'transformer': 'Transformer',
    'bdt': 'BDT',
    'depart_100M': 'iDeParT (100M jets)',
}


METRIC_NAMING_SCHEMA = {
    'eff_tag_efficiency': r'$\varepsilon_{\mathrm{eff}}$',
    'binary_accuracy': 'Accuracy',
    'relative_accuracy': 'Accuracy',
    'relative_error': 'Relative Error',
    'heatmap': 'Relative Error',
    'auc': 'AUC',
    'gluon_efficiency': r'$\varepsilon_g$',
    'quark_efficiency': r'$\varepsilon_q$',
    'gluon_rejection': r'$\varphi_g$',
    'quark_rejection': r'$\varphi_q$',
    'gluon_rej_at_quark_eff_0.9': r'$\varphi_g @_{0.9} \varepsilon_q$',
    'quark_rej_at_gluon_eff_0.9': r'$\varphi_q @_{0.9} \varepsilon_g$',
    'loss': 'Loss',
    'roc': 'ROC',
}

MC_NAMING_SCHEMA = {
    'pythia': 'Pythia',
    'sherpa': 'Sherpa',
    'sherpa_lund': 'Sherpa (Lund)',
    'herwig': 'Herwig7',
    'herwig_dipole': 'Herwig7 (Dipole)',
}
