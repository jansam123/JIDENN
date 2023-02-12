import tensorflow as tf
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional
#
from .utils.transformations import to_e_px_py_pz


ROOTVariables = Dict[str, Union[tf.RaggedTensor, tf.Tensor]]
JIDENNVariables = Dict[str, ROOTVariables]


@tf.function
def pick_PFO_kinematics(sample: JIDENNVariables) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    m = sample['perJetTuple']['jets_PFO_m']
    pt = sample['perJetTuple']['jets_PFO_pt']
    eta = sample['perJetTuple']['jets_PFO_eta']
    phi = sample['perJetTuple']['jets_PFO_phi']
    return m, pt, eta, phi


@tf.function
def pick_jet_kinematics(sample: JIDENNVariables) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    m = sample['perJet']['jets_m']
    pt = sample['perJet']['jets_pt']
    eta = sample['perJet']['jets_eta']
    phi = sample['perJet']['jets_phi']
    return m, pt, eta, phi


@tf.function
def PFO_interactions(sample: JIDENNVariables) -> ROOTVariables:
    m, pt, eta, phi = pick_PFO_kinematics(sample)
    E, px, py, pz = tf.unstack(to_e_px_py_pz(tf.stack([m, pt, eta, phi], axis=-1)), axis=-1)
    delta = tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                         tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
    k_t = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) * delta
    z = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) / (pt[:, tf.newaxis] + pt[tf.newaxis, :])
    m2 = tf.math.square(E[:, tf.newaxis] + E[tf.newaxis, :]) - tf.math.square(px[:, tf.newaxis] + px[tf.newaxis, :]) - \
        tf.math.square(py[:, tf.newaxis] + py[tf.newaxis, :]) - tf.math.square(pz[:, tf.newaxis] + pz[tf.newaxis, :])
    delta = tf.math.log(delta)
    delta = tf.linalg.set_diag(delta, tf.zeros_like(m))
    k_t = tf.math.log(k_t)
    k_t = tf.linalg.set_diag(k_t, tf.zeros_like(m))
    z = tf.linalg.set_diag(z, tf.zeros_like(m))
    m2 = tf.math.log(m2)
    m2 = tf.linalg.set_diag(m2, tf.zeros_like(m))
    return {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}


@tf.function
def PGOs_variables(sample: JIDENNVariables) -> ROOTVariables:
    PFO_m, PFO_pt, PFO_eta, PFO_phi = pick_PFO_kinematics(sample)
    jet_m, jet_pt, jet_eta, jet_phi = pick_jet_kinematics(sample)
    PFO_E = tf.math.sqrt(PFO_pt**2 + PFO_m**2)
    jet_E = tf.math.sqrt(jet_pt**2 + jet_m**2)
    deltaEta = PFO_eta - jet_eta
    deltaPhi = PFO_phi - jet_phi
    deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

    logPT = tf.math.log(PFO_pt)

    logPT_PTjet = tf.math.log(PFO_pt/jet_pt)
    logE = tf.math.log(PFO_E)
    logE_Ejet = tf.math.log(PFO_E/jet_E)
    m = PFO_m
    # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
    data = {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
            'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}
    return data


@tf.function
def PFOs_and_PFO_interactions_variables(sample: JIDENNVariables) -> Tuple[ROOTVariables, ROOTVariables]:
    interaction = PFO_interactions(sample)
    variables = PGOs_variables(sample)
    return variables, interaction


@tf.function
def jet_variables(sample: JIDENNVariables) -> ROOTVariables:
    data = {var: tf.cast(sample['perJet'][var], tf.float32) for var in sample['perJet'].keys()}
    data.update({var: tf.cast(sample['perEvent'][var], tf.float32) for var in sample['perEvent'].keys()})
    return data


@tf.function
def bdt_variables(sample: JIDENNVariables) -> ROOTVariables:
    m_jet, pt_jet, eta_jet, phi_jet = pick_jet_kinematics(sample)
    m_const, pt_const, eta_const, phi_const = pick_PFO_kinematics(sample)
    N_PFO = sample['perJet']['jets_PFO_n']
    delta_R_PFO_jet = tf.math.sqrt(tf.math.square(eta_jet - eta_const) +
                                   tf.math.square(phi_jet - phi_const))

    W_PFO_jet = tf.math.reduce_sum(pt_const * delta_R_PFO_jet, axis=-1)/tf.math.reduce_sum(pt_const, axis=-1)
    delta_R_PFOs = tf.math.sqrt(tf.math.square(
        eta_const[:, tf.newaxis] - eta_const[tf.newaxis, :]) + tf.math.square(phi_const[:, tf.newaxis] - phi_const[tf.newaxis, :]))
    C1_PFO_jet = tf.einsum('i,ij,j', pt_const, tf.linalg.set_diag(
        delta_R_PFOs, tf.zeros_like(pt_const))**0.2, pt_const) / tf.math.reduce_sum(pt_const, axis=-1)**2
    output_data = {'pt_jet': pt_jet, 'eta_jet': eta_jet, 'N_PFO': N_PFO,
                   'W_PFO_jet': W_PFO_jet, 'C1_PFO_jet': C1_PFO_jet}
    return output_data


def get_train_input(model: str, interaction: Optional[bool] = False) -> Callable:
    high_level_models = ['basic_fc', 'highway']
    PFO_based_models = ['transformer', 'part', 'depart']
    bdt_models = ['bdt']

    if model in high_level_models:
        return jet_variables
    elif model in PFO_based_models:
        if interaction:
            return PFOs_and_PFO_interactions_variables
        else:
            return PGOs_variables
    elif model in bdt_models:
        return bdt_variables
    else:
        raise ValueError(f"Model {model} not supported pick from {high_level_models + PFO_based_models + bdt_models}")

def get_input_shape(model: str, total_variables, interaction: Optional[bool] = False) -> Union[int, Tuple[Union[int, None]]]:
    high_level_models = ['basic_fc', 'highway']
    PFO_based_models = ['transformer', 'part', 'depart']
    bdt_models = ['bdt']

    if model in high_level_models:
        return total_variables
    elif model in PFO_based_models:
        if interaction:
            return ((None, 8), (None, None, 4))
        else:
            return (None, 8)
    elif model in bdt_models:
        return 5
    else:
        raise ValueError(f"Model {model} not supported pick from {high_level_models + PFO_based_models + bdt_models}")