"""
Module containing classes that create the various types of input variables for all the neural networks.
Each type of input variables is a subclass of the `TrainInput` class.
Defining a new type of input variables is as simple as creating a new subclass of `TrainInput` ,
implementing the `__call__` method and the `input_shape` property, which is used to define the 
input layer size of the neural network.
"""
import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional, Type
#
from .four_vector_transform import to_e_px_py_pz, to_m_pt_eta_phi
from .JIDENNDataset import ROOTVariables


class TrainInput(ABC):
    """Base class for all train input classes. The `TrainInput` class is used to **construct the input variables** for the neural network.
    The class can be initialized with a list of available variables. 

    The instance is then passed to the `map` method of a `tf.data.Dataset` object to use the `TrainInput.__call__` method to construct the input variables.
    Optionally, the class can be put into a `tf.function` to speed up the preprocessing.

    Example:
    ```python
    train_input = HighLevelJetVariables(variables=['jets_pt', 'jets_eta', 'jets_phi', 'jets_m'])
    dataset = dataset.map(tf.function(func=train_input))
    ```

    Args:
        variables (List[str], optional): List of available variables. Defaults to None.

    """

    def __init__(self, variables: Optional[List[str]] = None):
        self.variables = variables

    @abstractproperty
    def input_shape(self) -> Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]:
        """The shape of the input variables. This is used to **define the input layer size** of the neural network.
        The `None` values are used for ragged dimensions., eg. `(None, 4)` for a variable number of jet consitutents with 4 variables per consitutent.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        """Constructs the input variables from the `ROOTVariables` object. 
        The output is a dictionary of the form `{'var_name': tf.Tensor}`, i.e. a `ROOTVariables` type.

        Args:
            sample (ROOTVariables): The input sample.    

        Returns:
            ROOTVariables: The output variables of the form `{'var_name': tf.Tensor}`.

        """
        raise NotImplementedError


class HighLevelJetVariables(TrainInput):
    """Constructs the input variables characterizing the **whole jet**. 
    The variables are taken from the `variable` list on the input.
    These variables are used to train `jidenn.models.FC.FCModel` and `jidenn.models.Highway.HighwayModel`.

    Args:
        variables (List[str], optional): List of available variables. Defaults to None.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variables is None:
            self.variables = [
                'jets_ActiveArea4vec_eta',
                'jets_ActiveArea4vec_m',
                'jets_ActiveArea4vec_phi',
                'jets_ActiveArea4vec_pt',
                'jets_DetectorEta',
                'jets_FracSamplingMax',
                'jets_FracSamplingMaxIndex',
                'jets_GhostMuonSegmentCount',
                'jets_JVFCorr',
                'jets_JetConstitScaleMomentum_eta',
                'jets_JetConstitScaleMomentum_m',
                'jets_JetConstitScaleMomentum_phi',
                'jets_JetConstitScaleMomentum_pt',
                'jets_JvtRpt',
                'jets_fJVT',
                'jets_passFJVT',
                'jets_passJVT',
                'jets_Timing',
                'jets_Jvt',
                'jets_EMFrac',
                'jets_Width',
                'jets_chf',
                'jets_eta',
                'jets_m',
                'jets_phi',
                'jets_pt',
                'jets_PFO_n',
                'jets_ChargedPFOWidthPt1000[0]',
                'jets_TrackWidthPt1000[0]',
                'jets_NumChargedPFOPt1000[0]',
                'jets_SumPtChargedPFOPt500[0]',
                'jets_NumChargedPFOPt500[0]',
                'corrected_averageInteractionsPerCrossing[0]',
            ]

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        """Loops over the `per_jet_variables` and `per_event_variables` and constructs the input variables.

        Args:
            sample (ROOTVariables): The input sample.

        Returns:
            ROOTVariables: The output variables of the form `{'var_name': tf.Tensor}` where `var_name` is from `per_jet_variables` and `per_event_variables`.
        """

        return {var: tf.cast(sample[var], tf.float32) for var in self.variables}

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `len(self.variables)`."""
        return len(self.variables)


class QR(TrainInput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variables is None:
            self.variables = [
                'Electron',
                'IntermediateElectron',
                'Muon',
                'IntermediateMuon',
                'KinLepton',
                'IntermediateKinLepton',
                'Kaon',
                'SlowPion',
                'FastHadron',
                'Lambda',
                'FSC',
                'MaximumPstar',
                'KaonPion',
                'dx',
                'dy',
                'dz',
                'E',
                'charge',
                'px_c',
                'py_c',
                'pz_c',
                'electronID_c',
                'muonID_c',
                'pionID_c',
                'kaonID_c',
                'protonID_c',
                'deuteronID_c',
                'electronID_noSVD_noTOP_c',
            ]

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        data = {var: tf.cast(sample[var], tf.float32)
                for var in self.variables}
        px, py, pz, e = data.pop('px_c'), data.pop(
            'py_c'), data.pop('pz_c'), data.pop('E')
        p_norm = tf.norm(tf.stack([px, py, pz], axis=1), axis=1)
        phi = tf.math.atan2(py, px)
        theta = tf.math.atan2(tf.norm(tf.stack([px, py], axis=1), axis=1), pz)

        data.update({'log_pt': tf.math.log(p_norm), 'theta': theta,
                    'phi': phi, 'log_e': tf.math.log(e)})
        return data

    @property
    def input_shape(self) -> Tuple[None, int]:
        """The input shape is tuple `(None, len(per_jet_tuple_variables))`."""
        return (None, len(self.variables))


class QRInteraction(TrainInput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variables is None:
            self.variables = [
                'Electron',
                'IntermediateElectron',
                'Muon',
                'IntermediateMuon',
                'KinLepton',
                'IntermediateKinLepton',
                'Kaon',
                'SlowPion',
                'FastHadron',
                'Lambda',
                'FSC',
                'MaximumPstar',
                'KaonPion',
                'dx',
                'dy',
                'dz',
                'E',
                'charge',
                'px_c',
                'py_c',
                'pz_c',
                'electronID_c',
                'muonID_c',
                'pionID_c',
                'kaonID_c',
                'protonID_c',
                'deuteronID_c',
                'electronID_noSVD_noTOP_c',
            ]

    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        data = {var: tf.cast(sample[var], tf.float32)
                for var in self.variables}
        px, py, pz, e = data.pop('px_c'), data.pop(
            'py_c'), data.pop('pz_c'), data.pop('E')
        p_norm = tf.norm(tf.stack([px, py, pz], axis=1), axis=1)
        phi = tf.math.atan2(py, px)
        theta = tf.math.atan2(tf.norm(tf.stack([px, py], axis=1), axis=1), pz)

        data.update({'log_norm_p': tf.math.log(p_norm), 'theta': theta,
                    'phi': phi, 'log_e': tf.math.log(e)})

        delta = tf.math.sqrt(tf.math.square(theta[:, tf.newaxis] - theta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        delta = tf.math.log(delta)
        delta = tf.linalg.set_diag(delta, tf.zeros_like(e))

        k_t = tf.math.minimum(
            p_norm[:, tf.newaxis], p_norm[tf.newaxis, :]) * delta
        k_t = tf.math.log(k_t)
        k_t = tf.linalg.set_diag(k_t, tf.zeros_like(e))

        z = tf.math.minimum(p_norm[:, tf.newaxis], p_norm[tf.newaxis, :]) / \
            (p_norm[:, tf.newaxis] + p_norm[tf.newaxis, :])
        z = tf.linalg.set_diag(z, tf.zeros_like(e))

        m2 = tf.math.square(e[:, tf.newaxis] + e[tf.newaxis, :]) - tf.math.square(px[:, tf.newaxis] + px[tf.newaxis, :]) - \
            tf.math.square(py[:, tf.newaxis] + py[tf.newaxis, :]) - \
            tf.math.square(pz[:, tf.newaxis] + pz[tf.newaxis, :])
        m2 = tf.linalg.set_diag(m2, tf.zeros_like(e))
        m2 = tf.math.log(m2)

        interaction_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}

        # dx, dy, dz = data['dx'], data['dy'], data['dz']
        # poca_norm = tf.norm(tf.stack([dx, dy, dz], axis=1), axis=1)
        # delta_poca = tf.math.sqrt(tf.math.square(dx[:, tf.newaxis] - dx[tf.newaxis, :]) +
        #                           tf.math.square(dy[:, tf.newaxis] - dy[tf.newaxis, :]) +
        #                           tf.math.square(dz[:, tf.newaxis] - dz[tf.newaxis, :]))
        # delta_poca = tf.math.log(delta_poca)
        # delta_poca = tf.linalg.set_diag(delta_poca, tf.zeros_like(e))

        # k_t_poca = tf.math.minimum(
        #     poca_norm[:, tf.newaxis], poca_norm[tf.newaxis, :]) * delta_poca
        # k_t_poca = tf.math.log(k_t_poca)
        # k_t_poca = tf.linalg.set_diag(k_t_poca, tf.zeros_like(e))

        # z_poca = tf.math.minimum(poca_norm[:, tf.newaxis], poca_norm[tf.newaxis, :]) / \
        #     (poca_norm[:, tf.newaxis] + poca_norm[tf.newaxis, :])
        # z_poca = tf.linalg.set_diag(z_poca, tf.zeros_like(e))

        # interaction_vars.update(
        #     {'delta_poca': delta_poca, 'k_t_poca': k_t_poca, 'z_poca': z_poca})

        return data, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, None, int]]:
        return (None, len(self.variables)), (None, None, 4)


class HighLevelPFOVariables(TrainInput):
    """Constructs the input variables characterizing the **whole jet** from the PFO objects.
    These are special variables constructed for the BDT model, `jidenn.models.BDT.bdt_model`, from PFO object (originaly only from tracks trk)

    ##Variables: 
    - jet transverse momentum $$p_{\\mathrm{T}}^{\\mathrm{jet}}$$
    - jet psedo-rapidity $$\\eta^{\\mathrm{jet}}$$
    - number of PFOs $$ N_{\\mathrm {PFO}}=\\sum_{\\mathrm {PFO } \\in \\mathrm { jet }} $$
    - jet width $$$W_{\\mathrm {PFO}}=\\frac{\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a} \\sqrt{(\\eta^a - \\eta^{\\mathrm{jet}})^2 + (\\phi^a - \\phi^{\\mathrm{jet}})^2}}{\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a}}$$
    - C variable $$C_1^{\\beta=0.2}=\\frac{\\sum_{a, b \\in \\mathrm{jet}}^{a \\neq b} p_{\\mathrm{T}}^a p_{\\mathrm{T}}^b \\left(\\sqrt{(\\eta^a - \\eta^b)^2 + (\\phi^a - \\phi^b)^2}\\right)^{\\beta=0.2}}{\\left(\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a}\\right)^2}$$
    """

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        pt_jet = sample['jets_pt']
        eta_jet = sample['jets_eta']
        phi_jet = sample['jets_phi']

        pt_const = sample['jets_PFO_pt']
        eta_const = sample['jets_PFO_eta']
        phi_const = sample['jets_PFO_phi']

        N_PFO = sample['jets_PFO_n']

        delta_R_PFO_jet = tf.math.sqrt(tf.math.square(eta_jet - eta_const) +
                                       tf.math.square(phi_jet - phi_const))

        W_PFO_jet = tf.math.reduce_sum(
            pt_const * delta_R_PFO_jet, axis=-1) / tf.math.reduce_sum(pt_const, axis=-1)
        delta_R_PFOs = tf.math.sqrt(tf.math.square(
            eta_const[:, tf.newaxis] - eta_const[tf.newaxis, :]) + tf.math.square(phi_const[:, tf.newaxis] - phi_const[tf.newaxis, :]))
        C1_PFO_jet = tf.einsum('i,ij,j', pt_const, tf.linalg.set_diag(
            delta_R_PFOs, tf.zeros_like(pt_const))**0.2, pt_const) / tf.math.reduce_sum(pt_const, axis=-1)**2

        output_data = {'pt_jet': pt_jet, 'eta_jet': eta_jet, 'N_PFO': N_PFO,
                       'W_PFO_jet': W_PFO_jet, 'C1_PFO_jet': C1_PFO_jet}
        return output_data

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `5`, number of variables."""
        return 5


class ConstituentVariables(TrainInput):
    """Constructs the input variables characterizing the individual **jet constituents**, the PFO objects.
    These variables are used to train `jidenn.models.PFN.PFNModel`, `jidenn.models.EFN.EFNModel`, 
    `jidenn.models.Transformer.TransformerModel`, `jidenn.models.ParT.ParTModel`, `jidenn.models.DeParT.DeParTModel`.

    ##Variables: 
    - log of the constituent transverse momentum $$\\log(p_{\\mathrm{T}})$$
    - log of the constituent energy $$\\log(E)$$
    - mass of the constituent $$m$$
    - log of the fraction of the constituent energy to the jet energy $$\\log(E_{\\mathrm{const}}/E_{\\mathrm{jet}})$$
    - log of the fraction of the constituent transverse momentum to the jet transverse momentum $$\\log(p_{\\mathrm{T}}^{\\mathrm{const}}/p_{\\mathrm{T}}^{\\mathrm{jet}})$$
    - difference in the constituent and jet pseudorapidity $$\\Delta \\eta = \\eta^{\\mathrm{const}} - \\eta^{\\mathrm{jet}}$$
    - difference in the constituent and jet azimuthal angle $$\\Delta \\phi = \\phi^{\\mathrm{const}} - \\phi^{\\mathrm{jet}}$$
    - angular distance between the constituent and jet $$\\Delta R = \\sqrt{(\\Delta \\eta)^2 + (\\Delta \\phi)^2}$$
    """

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        m_const = sample['jets_PFO_m']
        pt_const = sample['jets_PFO_pt']
        eta_const = sample['jets_PFO_eta']
        phi_const = sample['jets_PFO_phi']

        m_jet = sample['jets_m']
        pt_jet = sample['jets_pt']
        eta_jet = sample['jets_eta']
        phi_jet = sample['jets_phi']

        PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta_const - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi_const - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(pt_const)

        logE = tf.math.log(PFO_E)
        logPT_PTjet = tf.math.log(pt_const / tf.math.reduce_mean(pt_jet))
        logE_Ejet = tf.math.log(PFO_E / tf.math.reduce_mean(jet_E))
        m = m_const
        # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
        return {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}

    @property
    def input_shape(self) -> Tuple[None, int]:
        """The input shape is `(None, 8)`, where `None` indicates that the number of constituents is not fixed, 
        and `8` is the number of variables per constituent."""
        return (None, 8)


class RelativeConstituentVariables(TrainInput):
    """Constructs the input variables characterizing the individual **jet constituents**, the PFO objects.
    It is the same as `ConstituentVariables` but containg only variables relative to the jet.
    These are used as alternative input to models mentioned in `ConstituentVariables`.

    ##Variables: 
    - mass of the constituent $$m$$
    - log of the fraction of the constituent energy to the jet energy $$\\log(E_{\\mathrm{const}}/E_{\\mathrm{jet}})$$
    - log of the fraction of the constituent transverse momentum to the jet transverse momentum $$\\log(p_{\\mathrm{T}}^{\\mathrm{const}}/p_{\\mathrm{T}}^{\\mathrm{jet}})$$
    - difference in the constituent and jet pseudorapidity $$\\Delta \\eta = \\eta^{\\mathrm{const}} - \\eta^{\\mathrm{jet}}$$
    - difference in the constituent and jet azimuthal angle $$\\Delta \\phi = \\phi^{\\mathrm{const}} - \\phi^{\\mathrm{jet}}$$
    - angular distance between the constituent and jet $$\\Delta R = \\sqrt{(\\Delta \\eta)^2 + (\\Delta \\phi)^2}$$
    """

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        m_const = sample['jets_PFO_m']
        pt_const = sample['jets_PFO_pt']
        eta_const = sample['jets_PFO_eta']
        phi_const = sample['jets_PFO_phi']

        m_jet = sample['jets_m']
        pt_jet = sample['jets_pt']
        eta_jet = sample['jets_eta']
        phi_jet = sample['jets_phi']

        PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta_const - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi_const - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(pt_const)

        logPT_PTjet = tf.math.log(pt_const / tf.math.reduce_mean(pt_jet))
        logE_Ejet = tf.math.log(PFO_E / tf.math.reduce_mean(jet_E))
        logE = tf.math.log(PFO_E)
        m = m_const
        # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
        return {'log_PT|PTjet': logPT_PTjet, 'log_E|Ejet': logE_Ejet,
                'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}

    @property
    def input_shape(self) -> Tuple[None, int]:
        """The input shape is `(None, 6)`, where `None` indicates that the number of constituents is not fixed, 
        and `6` is the number of variables per constituent."""
        return (None, 6)


class InteractionConstituentVariables(TrainInput):
    """Constructs the input variables characterizing the individual **jet constituents**, but on top of the
    `ConstituentVariables` it also includes the interaction variables, i.e. the variables characterizing the
    pair of constituents.
    These are used in the `jidenn.models.ParT.ParTModel`, `jidenn.models.DeParT.DeParTModel`.

    ##Variables: 
    ###Constituent variables:
    - log of the constituent transverse momentum $$\\log(p_{\\mathrm{T}})$$
    - log of the constituent energy $$\\log(E)$$
    - mass of the constituent $$m$$
    - log of the fraction of the constituent energy to the jet energy $$\\log(E_{\\mathrm{const}}/E_{\\mathrm{jet}})$$
    - log of the fraction of the constituent transverse momentum to the jet transverse momentum $$\\log(p_{\\mathrm{T}}^{\\mathrm{const}}/p_{\\mathrm{T}}^{\\mathrm{jet}})$$
    - difference in the constituent and jet pseudorapidity $$\\Delta \\eta = \\eta^{\\mathrm{const}} - \\eta^{\\mathrm{jet}}$$
    - difference in the constituent and jet azimuthal angle $$\\Delta \\phi = \\phi^{\\mathrm{const}} - \\phi^{\\mathrm{jet}}$$
    - angular distance between the constituent and jet $$\\Delta R = \\sqrt{(\\Delta \\eta)^2 + (\\Delta \\phi)^2}$$
    ###Interaction variables:
    - log of the angular distance between the constituents $$\\log \\Delta  = \\sqrt{(\\eta^a - \\eta^b)^2 + (\\phi^a - \\phi^b)^2}$$
    - log of the kt variable $$\\log k_\\mathrm{T} = \\log \\mathrm{min}(p_{\\mathrm{T}}^a, p_{\\mathrm{T}}^b) \\Delta $$
    - the fraction of carried transverse momentum of the softer constituent $$z = \\frac{\\mathrm{min}(p_{\\mathrm{T}}^a, p_{\\mathrm{T}}^b)}{p_{\\mathrm{T}}^a + p_{\\mathrm{T}}^b}$$
    - the log of invariant mass $$\\log m^2 = \\log{(p^{\\mu, a} + p^{\\mu, b})^2}$$

    """

    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        m = sample['jets_PFO_m']
        pt = sample['jets_PFO_pt']
        eta = sample['jets_PFO_eta']
        phi = sample['jets_PFO_phi']

        E, px, py, pz = tf.unstack(to_e_px_py_pz(
            tf.stack([m, pt, eta, phi], axis=-1)), axis=-1)
        delta = tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        delta = tf.math.log(delta)
        delta = tf.linalg.set_diag(delta, tf.zeros_like(m))

        k_t = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) * delta
        k_t = tf.math.log(k_t)
        k_t = tf.linalg.set_diag(k_t, tf.zeros_like(m))

        z = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) / \
            (pt[:, tf.newaxis] + pt[tf.newaxis, :])
        z = tf.linalg.set_diag(z, tf.zeros_like(m))

        m2 = tf.math.square(E[:, tf.newaxis] + E[tf.newaxis, :]) - tf.math.square(px[:, tf.newaxis] + px[tf.newaxis, :]) - \
            tf.math.square(py[:, tf.newaxis] + py[tf.newaxis, :]) - \
            tf.math.square(pz[:, tf.newaxis] + pz[tf.newaxis, :])
        m2 = tf.linalg.set_diag(m2, tf.zeros_like(m))
        m2 = tf.math.log(m2)

        interaction_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}

        m_jet = sample['jets_m']
        pt_jet = sample['jets_pt']
        eta_jet = sample['jets_eta']
        phi_jet = sample['jets_phi']

        PFO_E = tf.math.sqrt(pt**2 + m**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(pt)

        logPT_PTjet = tf.math.log(pt / tf.math.reduce_sum(pt_jet))
        logE = tf.math.log(PFO_E)
        logE_Ejet = tf.math.log(PFO_E / tf.math.reduce_mean(jet_E))

        const_vars = {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                      'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}

        return const_vars, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, None, int]]:
        """The input shape is a tuple of two tuples `(None, 8)` and `(None, None, 4)`, where the first tuple corresponds to the shape 
        of the variables for each constituent and the second tuple corresponds to the shape of a variable for each pair of constituents,
        i.e. a matrix for each jet."""
        return (None, 8), (None, None, 4)


def input_classes_lookup(class_name: Literal['highlevel',
                                             'highlevel_constituents',
                                             'constituents',
                                             'relative_constituents',
                                             'interaction_constituents']) -> Type[TrainInput]:
    """Function to return the input class based on the class name.
    It is used to pick training class based on the config file.

    Args:
        class_name (str): Name of the class to return. Options are: 'highlevel', 'highlevel_constituents', 'constituents', 'relative_constituents', 'interaction_constituents'

    Raises:
        ValueError: If the class name is not in the list of options.

    Returns:
        Type[TrainInput]: The class to use for training. **Not as instance**, but as class itself.
    """

    lookup_dict = {'highlevel': HighLevelJetVariables,
                   'highlevel_constituents': HighLevelPFOVariables,
                   'constituents': ConstituentVariables,
                   'relative_constituents': RelativeConstituentVariables,
                   'interaction_constituents': InteractionConstituentVariables,
                   'qr': QR,
                   'qr_interaction': QRInteraction}

    if class_name not in lookup_dict.keys():
        raise ValueError(f'Unknown input class name {class_name}')

    return lookup_dict[class_name]


__pdoc__ = {f'{local_class.__name__}.__call__':
            True for local_class in TrainInput.__subclasses__() + [TrainInput]}
