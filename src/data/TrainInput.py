import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional, Type
import src.config.config_subclasses as cfg
#
from .utils.transformations import to_e_px_py_pz


ROOTVariables = Dict[str, Union[tf.Tensor, tf.RaggedTensor]]
JIDENNVariables = Dict[Literal['perJet', 'perJetTuple', 'perEvent'], ROOTVariables]


class TrainInput(ABC):
    def __init__(self, per_jet_variables: Optional[List[str]] = None,
                 per_event_variables: Optional[List[str]] = None,
                 per_jet_tuple_variables: Optional[List[str]] = None):

        self.per_jet_variables = per_jet_variables
        self.per_event_variables = per_event_variables
        self.per_jet_tuple_variables = per_jet_tuple_variables

    @abstractproperty
    def input_shape(self) -> Union[int, Tuple[Union[None, int]]]:
        pass

    @abstractmethod
    def __call__(self, sample: JIDENNVariables) -> ROOTVariables:
        pass


class HighLevelJetVariables(TrainInput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.per_jet_variables is None:
            raise ValueError("per_jet_variables must be specified")

    def __call__(self, sample: JIDENNVariables) -> ROOTVariables:
        data = {var: tf.cast(sample['perJet'][var], tf.float32) for var in self.per_jet_variables}
        if self.per_event_variables is None:
            return data
        data.update({var: tf.cast(sample['perEvent'][var], tf.float32) for var in self.per_event_variables})
        return data

    @property
    def input_shape(self) -> int:
        return len(self.per_jet_variables) + (len(self.per_event_variables) if self.per_event_variables is not None else 0)


class HighLevelPFOVariables(TrainInput):

    def __call__(self, sample: JIDENNVariables) -> ROOTVariables:
        pt_jet = sample['perJet']['jets_pt']
        eta_jet = sample['perJet']['jets_eta']
        phi_jet = sample['perJet']['jets_phi']

        pt_const = sample['perJetTuple']['jets_PFO_pt']
        eta_const = sample['perJetTuple']['jets_PFO_eta']
        phi_const = sample['perJetTuple']['jets_PFO_phi']

        N_PFO = sample['perJet']['jets_PFO_n']

        delta_R_PFO_jet = tf.math.sqrt(tf.math.square(eta_jet - eta_const) +
                                       tf.math.square(phi_jet - phi_const))

        W_PFO_jet = tf.math.reduce_sum(pt_const * delta_R_PFO_jet, axis=-1) / tf.math.reduce_sum(pt_const, axis=-1)
        delta_R_PFOs = tf.math.sqrt(tf.math.square(
            eta_const[:, tf.newaxis] - eta_const[tf.newaxis, :]) + tf.math.square(phi_const[:, tf.newaxis] - phi_const[tf.newaxis, :]))
        C1_PFO_jet = tf.einsum('i,ij,j', pt_const, tf.linalg.set_diag(
            delta_R_PFOs, tf.zeros_like(pt_const))**0.2, pt_const) / tf.math.reduce_sum(pt_const, axis=-1)**2

        output_data = {'pt_jet': pt_jet, 'eta_jet': eta_jet, 'N_PFO': N_PFO,
                       'W_PFO_jet': W_PFO_jet, 'C1_PFO_jet': C1_PFO_jet}
        return output_data

    @property
    def input_shape(self) -> int:
        return 5


class ConstituentVariables(TrainInput):

    def __call__(self, sample: JIDENNVariables) -> ROOTVariables:
        m_const = sample['perJetTuple']['jets_PFO_m']
        pt_const = sample['perJetTuple']['jets_PFO_pt']
        eta_const = sample['perJetTuple']['jets_PFO_eta']
        phi_const = sample['perJetTuple']['jets_PFO_phi']

        m_jet = sample['perJet']['jets_m']
        pt_jet = sample['perJet']['jets_pt']
        eta_jet = sample['perJet']['jets_eta']
        phi_jet = sample['perJet']['jets_phi']

        PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta_const - eta_jet
        deltaPhi = phi_const - phi_jet
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(pt_const)

        logPT_PTjet = tf.math.log(pt_const / pt_jet)
        logE = tf.math.log(PFO_E)
        logE_Ejet = tf.math.log(PFO_E / jet_E)
        m = m_const
        # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
        return {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}

    @property
    def input_shape(self) -> Tuple[None, int]:
        return (None, 8)


class RelativeConstituentVariables(TrainInput):

    def __call__(self, sample: JIDENNVariables) -> ROOTVariables:
        m_const = sample['perJetTuple']['jets_PFO_m']
        pt_const = sample['perJetTuple']['jets_PFO_pt']
        eta_const = sample['perJetTuple']['jets_PFO_eta']
        phi_const = sample['perJetTuple']['jets_PFO_phi']

        m_jet = sample['perJet']['jets_m']
        pt_jet = sample['perJet']['jets_pt']
        eta_jet = sample['perJet']['jets_eta']
        phi_jet = sample['perJet']['jets_phi']

        PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta_const - eta_jet
        deltaPhi = phi_const - phi_jet
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(pt_const)

        logPT_PTjet = tf.math.log(pt_const / pt_jet)
        logE = tf.math.log(PFO_E)
        logE_Ejet = tf.math.log(PFO_E / jet_E)
        m = m_const
        # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
        return {'log_PT|PTjet': logPT_PTjet, 'log_E|Ejet': logE_Ejet,
                'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}

    @property
    def input_shape(self) -> Tuple[None, int]:
        return (None, 6)


class InteractionConstituentVariables(TrainInput):

    def __call__(self, sample: JIDENNVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        m = sample['perJetTuple']['jets_PFO_m']
        pt = sample['perJetTuple']['jets_PFO_pt']
        eta = sample['perJetTuple']['jets_PFO_eta']
        phi = sample['perJetTuple']['jets_PFO_phi']

        E, px, py, pz = tf.unstack(to_e_px_py_pz(tf.stack([m, pt, eta, phi], axis=-1)), axis=-1)
        delta = tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        k_t = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) * delta
        z = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) / (pt[:, tf.newaxis] + pt[tf.newaxis, :])
        m2 = tf.math.square(E[:, tf.newaxis] + E[tf.newaxis, :]) - tf.math.square(px[:, tf.newaxis] + px[tf.newaxis, :]) - \
            tf.math.square(py[:, tf.newaxis] + py[tf.newaxis, :]) - \
            tf.math.square(pz[:, tf.newaxis] + pz[tf.newaxis, :])
        delta = tf.math.log(delta)
        delta = tf.linalg.set_diag(delta, tf.zeros_like(m))
        k_t = tf.math.log(k_t)
        k_t = tf.linalg.set_diag(k_t, tf.zeros_like(m))
        z = tf.linalg.set_diag(z, tf.zeros_like(m))
        m2 = tf.math.log(m2)
        m2 = tf.linalg.set_diag(m2, tf.zeros_like(m))
        interaction_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}

        m_jet = sample['perJet']['jets_m']
        pt_jet = sample['perJet']['jets_pt']
        eta_jet = sample['perJet']['jets_eta']
        phi_jet = sample['perJet']['jets_phi']

        PFO_E = tf.math.sqrt(pt**2 + m**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta - eta_jet
        deltaPhi = phi - phi_jet
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(pt)

        logPT_PTjet = tf.math.log(pt / pt_jet)
        logE = tf.math.log(PFO_E)
        logE_Ejet = tf.math.log(PFO_E / jet_E)

        const_vars = {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                      'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}

        return const_vars, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, None, int]]:
        return (None, 8), (None, None, 4)


class DeepSetConstituentVariables(TrainInput):
    def __call__(self, sample: JIDENNVariables) -> ROOTVariables:
        pt_const = sample['perJetTuple']['jets_PFO_pt']
        eta_const = sample['perJetTuple']['jets_PFO_eta']
        phi_const = sample['perJetTuple']['jets_PFO_phi']

        # pt_jet = sample['perJet']['jets_pt']
        eta_jet = sample['perJet']['jets_eta']
        phi_jet = sample['perJet']['jets_phi']

        logPT_PTjet = - tf.math.log(pt_const / tf.math.reduce_sum(pt_const))

        eta = eta_const - eta_jet
        phi = phi_const - phi_jet

        return {'log_pT': logPT_PTjet, 'deltaEta': eta, 'deltaPhi': phi}

    @property
    def input_shape(self) -> Tuple[None, int]:
        return (None, 3)


def input_classes_lookup(class_name: str) -> Type[TrainInput]:

    lookup_dict = {'highlevel': HighLevelJetVariables,
                   'highlevel_constituents': HighLevelPFOVariables,
                   'constituents': ConstituentVariables,
                   'relative_constituents': RelativeConstituentVariables,
                   'interaction_constituents': InteractionConstituentVariables,
                   'deepset_constituents': DeepSetConstituentVariables}

    if class_name not in lookup_dict.keys():
        raise ValueError(f'Unknown input class name {class_name}')

    return lookup_dict[class_name]
