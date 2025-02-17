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
from .four_vector_transform import to_e_px_py_pz, to_m_pt_eta_phi, cal_delta_phi, to_m_pt_y_phi
from .JIDENNDataset import ROOTVariables


CLIP_MIN = 1e-10
CLIP_MAX = 1e10

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

    def __init__(self, variables: Optional[List[str]] = None, max_constituents: Optional[int] = 100, constituent_name: Literal['PFO', 'Constituent'] = 'Constituent'):
        self.variables = variables
        self.max_constituents = max_constituents
        self.const_name = constituent_name

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


class FullHighLevelJetVariables(TrainInput):
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
                'corrected_averageInteractionsPerCrossing',
                f'jets_{self.const_name}_n',]

            self.idxd_variables = ['jets_ChargedPFOWidthPt1000',
                                   'jets_TrackWidthPt1000',
                                   'jets_NumChargedPFOPt1000',
                                   'jets_SumPtChargedPFOPt500',
                                   'jets_NumChargedPFOPt500',]

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        """Loops over the `per_jet_variables` and `per_event_variables` and constructs the input variables.

        Args:
            sample (ROOTVariables): The input sample.

        Returns:
            ROOTVariables: The output variables of the form `{'var_name': tf.Tensor}` where `var_name` is from `per_jet_variables` and `per_event_variables`.
        """

        new_sample = {var: tf.cast(sample[var], tf.float32)
                      for var in self.variables}
        new_sample.update(
            {var: tf.cast(sample[var][0], tf.float32) for var in self.idxd_variables})
        return new_sample

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `len(self.variables)`."""
        return len(self.variables) + len(self.idxd_variables)


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
                'jets_EMFrac',
                'jets_chf',
                'jets_eta',
                'jets_m',
                'jets_phi',
                'jets_pt',
                'jets_Width',
                # f'jets_{self.const_name}_n',
            ]

            self.idxd_variables = [
                # 'jets_NumChargedPFOPt1000',
                'jets_NumChargedPFOPt500',
                # 'jets_ChargedPFOWidthPt1000',
            ]

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        """Loops over the `per_jet_variables` and `per_event_variables` and constructs the input variables.

        Args:
            sample (ROOTVariables): The input sample.

        Returns:
            ROOTVariables: The output variables of the form `{'var_name': tf.Tensor}` where `var_name` is from `per_jet_variables` and `per_event_variables`.
        """

        new_sample = {var: tf.cast(sample[var], tf.float32)
                      for var in self.variables}
        new_sample.update(
            {var: tf.cast(sample[var][0], tf.float32) for var in self.idxd_variables})
        return new_sample

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `len(self.variables)`."""
        return len(self.variables) + len(self.idxd_variables)

class HighLevelJetVariablesNoEta(TrainInput):
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
                'jets_EMFrac',
                'jets_chf',
                'jets_m',
                'jets_phi',
                'jets_pt',
                'jets_Width',
            ]

            self.idxd_variables = [
                'jets_NumChargedPFOPt500',
            ]

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        """Loops over the `per_jet_variables` and `per_event_variables` and constructs the input variables.

        Args:
            sample (ROOTVariables): The input sample.

        Returns:
            ROOTVariables: The output variables of the form `{'var_name': tf.Tensor}` where `var_name` is from `per_jet_variables` and `per_event_variables`.
        """

        new_sample = {var: tf.cast(sample[var], tf.float32)
                      for var in self.variables}
        new_sample.update(
            {var: tf.cast(sample[var][0], tf.float32) for var in self.idxd_variables})
        return new_sample

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `len(self.variables)`."""
        return len(self.variables) + len(self.idxd_variables)


class CraftedHighLevelJetVariables(TrainInput):
    """Constructs the input variables characterizing the **whole jet** from the PFO objects.
    These are special variables constructed for the BDT model, `jidenn.models.BDT.bdt_model`, from PFO object (originaly only from tracks trk)

    ##Variables: 
    - jet transverse momentum $$p_{\\mathrm{T}}^{\\mathrm{jet}}$$
    - jet psedo-rapidity $$\\eta^{\\mathrm{jet}}$$
    - number of PFOs $$ N_{\\mathrm {PFO}}=\\sum_{\\mathrm {PFO } \\in \\mathrm { jet }} $$
    - jet width $$$W_{\\mathrm {PFO}}=\\frac{\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a} \\sqrt{(\\eta^a - \\eta^{\\mathrm{jet}})^2 + (\\phi^a - \\phi^{\\mathrm{jet}})^2}}{\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a}}$$
    - C variable $$C_1^{\\beta=0.2}=\\frac{\\sum_{a, b \\in \\mathrm{jet}}^{a \\neq b} p_{\\mathrm{T}}^a p_{\\mathrm{T}}^b \\left(\\sqrt{(\\eta^a - \\eta^b)^2 + (\\phi^a - \\phi^b)^2}\\right)^{\\beta=0.2}}{\\left(\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a}\\right)^2}$$
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_name = 'Constituent'

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        pt_jet = sample['jets_pt']
        eta_jet = sample['jets_eta']
        phi_jet = sample['jets_phi']

        pt_const = sample[f'jets_{self.const_name}_pt']
        eta_const = sample[f'jets_{self.const_name}_eta']
        phi_const = sample[f'jets_{self.const_name}_phi']

        N_PFO = sample[f'jets_{self.const_name}_n']

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

class CraftedHighLevelJetVariablesNoEta(TrainInput):
    """Constructs the input variables characterizing the **whole jet** from the PFO objects.
    These are special variables constructed for the BDT model, `jidenn.models.BDT.bdt_model`, from PFO object (originaly only from tracks trk)

    ##Variables: 
    - jet transverse momentum $$p_{\\mathrm{T}}^{\\mathrm{jet}}$$
    - jet psedo-rapidity $$\\eta^{\\mathrm{jet}}$$
    - number of PFOs $$ N_{\\mathrm {PFO}}=\\sum_{\\mathrm {PFO } \\in \\mathrm { jet }} $$
    - jet width $$$W_{\\mathrm {PFO}}=\\frac{\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a} \\sqrt{(\\eta^a - \\eta^{\\mathrm{jet}})^2 + (\\phi^a - \\phi^{\\mathrm{jet}})^2}}{\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a}}$$
    - C variable $$C_1^{\\beta=0.2}=\\frac{\\sum_{a, b \\in \\mathrm{jet}}^{a \\neq b} p_{\\mathrm{T}}^a p_{\\mathrm{T}}^b \\left(\\sqrt{(\\eta^a - \\eta^b)^2 + (\\phi^a - \\phi^b)^2}\\right)^{\\beta=0.2}}{\\left(\\sum_{a \\in \\mathrm{jet}} p_{\\mathrm{T}}^{a}\\right)^2}$$
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_name = 'Constituent'

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        pt_jet = sample['jets_pt']
        eta_jet = sample['jets_eta']
        phi_jet = sample['jets_phi']

        pt_const = sample[f'jets_{self.const_name}_pt']
        eta_const = sample[f'jets_{self.const_name}_eta']
        phi_const = sample[f'jets_{self.const_name}_phi']

        N_PFO = sample[f'jets_{self.const_name}_n']

        delta_R_PFO_jet = tf.math.sqrt(tf.math.square(eta_jet - eta_const) +
                                       tf.math.square(phi_jet - phi_const))

        W_PFO_jet = tf.math.reduce_sum(
            pt_const * delta_R_PFO_jet, axis=-1) / tf.math.reduce_sum(pt_const, axis=-1)
        delta_R_PFOs = tf.math.sqrt(tf.math.square(
            eta_const[:, tf.newaxis] - eta_const[tf.newaxis, :]) + tf.math.square(phi_const[:, tf.newaxis] - phi_const[tf.newaxis, :]))
        C1_PFO_jet = tf.einsum('i,ij,j', pt_const, tf.linalg.set_diag(
            delta_R_PFOs, tf.zeros_like(pt_const))**0.2, pt_const) / tf.math.reduce_sum(pt_const, axis=-1)**2

        output_data = {'pt_jet': pt_jet, 'N_PFO': N_PFO,
                       'W_PFO_jet': W_PFO_jet, 'C1_PFO_jet': C1_PFO_jet}
        return output_data

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `5`, number of variables."""
        return 4
    
class ConstituentBase(TrainInput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jet_type = 'jets'
        self.const_types = [self.const_name]
        
    def get_constituents(self, sample: ROOTVariables) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        m_const = []
        pt_const = []
        eta_const = []
        phi_const = []
        info_index = []
        for i, const_type in enumerate(self.const_types):
            m_const.append(sample[f'{self.jet_type}_{const_type}_m'])
            pt_const.append(sample[f'{self.jet_type}_{const_type}_pt'])
            eta_const.append(sample[f'{self.jet_type}_{const_type}_eta'])
            phi_const.append(sample[f'{self.jet_type}_{const_type}_phi'])
            info_index.append(i * tf.ones_like(pt_const[-1]))

        m_const = tf.concat(m_const, axis=0)
        pt_const = tf.concat(pt_const, axis=0)
        eta_const = tf.concat(eta_const, axis=0)
        phi_const = tf.concat(phi_const, axis=0)
        info_index = tf.concat(info_index, axis=0)

        sort_idxs = tf.argsort(pt_const, axis=0, direction='DESCENDING')

        m_const = tf.gather(m_const, sort_idxs)
        pt_const = tf.gather(pt_const, sort_idxs)
        eta_const = tf.gather(eta_const, sort_idxs)
        phi_const = tf.gather(phi_const, sort_idxs)
        info_index = tf.gather(info_index, sort_idxs)

        if self.max_constituents is not None:
            m_const = m_const[..., :self.max_constituents]
            pt_const = pt_const[..., :self.max_constituents]
            eta_const = eta_const[..., :self.max_constituents]
            phi_const = phi_const[..., :self.max_constituents]
            info_index = info_index[..., :self.max_constituents]
        
        info_index = tf.cast(info_index, tf.int32)
        info_index = tf.one_hot(info_index, len(self.const_types)) if len(self.const_types) > 2 else info_index
        
        return m_const, pt_const, eta_const, phi_const, info_index

    def get_jet(self, sample: ROOTVariables):
        m_jet = sample[f'{self.jet_type}_m']
        pt_jet = sample[f'{self.jet_type}_pt']
        eta_jet = sample[f'{self.jet_type}_eta']
        phi_jet = sample[f'{self.jet_type}_phi']
        return m_jet, pt_jet, eta_jet, phi_jet

    def get_interacting_variables(self, m, pt, eta, phi, E, px, py, pz):
        unlog_delta = tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        delta = tf.math.log(tf.clip_by_value(unlog_delta, CLIP_MIN, CLIP_MAX))
        delta = tf.linalg.set_diag(delta, tf.zeros_like(m))

        k_t = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) * tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        k_t = tf.math.log(tf.clip_by_value(k_t, CLIP_MIN, CLIP_MAX))
        k_t = tf.linalg.set_diag(k_t, tf.zeros_like(m))

        z = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) / \
            (pt[:, tf.newaxis] + pt[tf.newaxis, :])
        z = tf.linalg.set_diag(z, tf.zeros_like(m))

        m2 = tf.math.square(E[:, tf.newaxis] + E[tf.newaxis, :]) - tf.math.square(px[:, tf.newaxis] + px[tf.newaxis, :]) - \
            tf.math.square(py[:, tf.newaxis] + py[tf.newaxis, :]) - \
            tf.math.square(pz[:, tf.newaxis] + pz[tf.newaxis, :])
        m2 = tf.math.log(tf.clip_by_value(m2, CLIP_MIN, CLIP_MAX))
        m2 = tf.linalg.set_diag(m2, tf.zeros_like(m))
        return delta, k_t, z, m2
    
    def get_interacting_variables2(self, m, pt, eta, phi, E_un, px_un, py_un, pz_un):
        unlog_delta = tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        delta = tf.math.log(tf.clip_by_value(unlog_delta, CLIP_MIN, CLIP_MAX))
        delta = tf.linalg.set_diag(delta, tf.zeros_like(m))

        k_t = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) * tf.math.sqrt(tf.math.square(eta[:, tf.newaxis] - eta[tf.newaxis, :]) +
                             tf.math.square(phi[:, tf.newaxis] - phi[tf.newaxis, :]))
        k_t = tf.math.log(tf.clip_by_value(k_t, CLIP_MIN, CLIP_MAX))
        k_t = tf.linalg.set_diag(k_t, tf.zeros_like(m))

        z = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) / \
            (pt[:, tf.newaxis] + pt[tf.newaxis, :])
        z = tf.linalg.set_diag(z, tf.zeros_like(m))

        m2 = tf.math.square(E_un[:, tf.newaxis] + E_un[tf.newaxis, :]) - tf.math.square(px_un[:, tf.newaxis] + px_un[tf.newaxis, :]) - \
            tf.math.square(py_un[:, tf.newaxis] + py_un[tf.newaxis, :]) - \
            tf.math.square(pz_un[:, tf.newaxis] + pz_un[tf.newaxis, :])
        m2 = tf.math.log(tf.clip_by_value(m2, CLIP_MIN, CLIP_MAX))
        m2 = tf.linalg.set_diag(m2, tf.zeros_like(m))
        return delta, k_t, z, m2


class ConstituentVariables(ConstituentBase):
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
        m_const, pt_const, eta_const, phi_const, info_idx = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)

        PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta_const - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi_const - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        
        logPT = tf.math.log(tf.clip_by_value(pt_const, CLIP_MIN, CLIP_MAX))
        logE = tf.math.log(tf.clip_by_value(PFO_E, CLIP_MIN, CLIP_MAX))
        logPT_PTjet = tf.math.log(tf.clip_by_value(pt_const / tf.math.reduce_mean(pt_jet), CLIP_MIN, CLIP_MAX))
        logE_Ejet = tf.math.log(tf.clip_by_value(PFO_E / tf.math.reduce_mean(jet_E), CLIP_MIN, CLIP_MAX))
        
        m = tf.math.log(tf.clip_by_value(m_const, CLIP_MIN, CLIP_MAX))
        
        
        const_var =  {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR, 'm': m}
        
        if len(self.const_types) > 2:
            for i, const_type in enumerate(self.const_types):
                const_var[f'is_{const_type}'] = info_idx[:, i]
        elif len(self.const_types) == 2:
            const_var[f'is_{self.const_types[1]}'] = info_idx
        
        return const_var

    @property
    def input_shape(self) -> Tuple[None, int]:
        """The input shape is `(None, 8)`, where `None` indicates that the number of constituents is not fixed, 
        and `8` is the number of variables per constituent."""
                
        if len(self.const_types) == 1:
            id_var = 0
        elif len(self.const_types) == 2:
            id_var = 1
        else:
            id_var = len(self.const_types)
            
        return (None, 8+id_var)
    
class RelativeConstituentVariables(ConstituentBase):
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
        m, pt, eta, phi, info_idx = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)

        E, px, py, pz = to_e_px_py_pz(m, pt, eta, phi)

        px_jet = tf.reduce_sum(px, axis=-1)
        py_jet = tf.reduce_sum(py, axis=-1)
        pz_jet = tf.reduce_sum(pz, axis=-1)
        
        pt_jet = tf.math.sqrt(px_jet**2 + py_jet**2)
        eta_jet = tf.math.asinh(pz_jet / pt_jet)
        phi_jet = tf.math.atan2(py_jet, px_jet)

        px = px / pt_jet
        py = py / pt_jet
        pz = pz / pt_jet
        
        E = E / pt_jet
        pt = pt / tf.math.reduce_mean(pt_jet)
        eta = eta - tf.math.reduce_mean(eta_jet)
        phi = phi - tf.math.reduce_mean(phi_jet)

        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT_PTjet = tf.math.log(tf.clip_by_value(pt / tf.math.reduce_mean(pt_jet), CLIP_MIN, CLIP_MAX))
        logE_Ejet = tf.math.log(tf.clip_by_value(E / tf.math.reduce_mean(jet_E), CLIP_MIN, CLIP_MAX))
        m = tf.math.log(tf.clip_by_value(m, CLIP_MIN, CLIP_MAX))
        
        const_var =  {'log_PT|PTjet': logPT_PTjet, 'log_E|Ejet': logE_Ejet,
                'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR, 'm': m}
        
        if len(self.const_types) > 2:
            for i, const_type in enumerate(self.const_types):
                const_var[f'is_{const_type}'] = info_idx[:, i]
        elif len(self.const_types) == 2:
            const_var[f'is_{self.const_types[1]}'] = info_idx
        
        return const_var

    @property
    def input_shape(self) -> Tuple[None, int]:
        """The input shape is `(None, 6)`, where `None` indicates that the number of constituents is not fixed, 
        and `6` is the number of variables per constituent."""
        if len(self.const_types) == 1:
            id_var = 0
        elif len(self.const_types) == 2:
            id_var = 1
        else:
            id_var = len(self.const_types) 
        return (None, 6+id_var)

class GNNVariables(ConstituentBase):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.max_constituents is None:
            raise ValueError('max_constituents must be set in GNNVariables')

    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        m_const, pt_const, eta_const, phi_const, info_idx = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)

        PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta_const - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi_const - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(tf.clip_by_value(pt_const, CLIP_MIN, CLIP_MAX))

        logE = tf.math.log(tf.clip_by_value(PFO_E, CLIP_MIN, CLIP_MAX))
        logPT_PTjet = tf.math.log(tf.clip_by_value(pt_const / tf.math.reduce_mean(pt_jet), CLIP_MIN, CLIP_MAX))
        logE_Ejet = tf.math.log(tf.clip_by_value(PFO_E / tf.math.reduce_mean(jet_E), CLIP_MIN, CLIP_MAX))
        m = tf.math.log(tf.clip_by_value(m_const, CLIP_MIN, CLIP_MAX))
        # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
        fts = {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR, 'm': m}
        
        if len(self.const_types) > 2:
            for i, const_type in enumerate(self.const_types):
                fts[f'is_{const_type}'] = info_idx[:, i]
        elif len(self.const_types) == 2:
            fts[f'is_{self.const_types[1]}'] = info_idx

        mask = tf.ones_like(fts['deltaEta'])
        mask = tf.pad(mask, [[0, self.max_constituents - tf.shape(mask)[-1]]])

        for fts_name, fts_value in fts.items():
            fts[fts_name] = tf.pad(
                fts_value, [[0, self.max_constituents - tf.shape(fts_value)[-1]]])

        points = {'deltaEta': fts['deltaEta'], 'deltaPhi': fts['deltaPhi']}
        mask = {'mask': mask}
        return points, fts, mask

    @property
    def input_shape(self) -> Tuple[None, int]:
        """The input shape is `(None, 8)`, where `None` indicates that the number of constituents is not fixed, 
        and `8` is the number of variables per constituent."""
        
        if len(self.const_types) == 1:
            id_var = 0
        elif len(self.const_types) == 2:
            id_var = 1
        else:
            id_var = len(self.const_types) 
            
        return (self.max_constituents, 2), (self.max_constituents, 8+id_var), (self.max_constituents, 1)


class IRCSVariables(ConstituentBase):
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

    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        m_const, pt_const, eta_const, phi_const, _ = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)

        deltaEta = eta_const - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi_const - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        PT_PTjet = pt_const / tf.math.reduce_mean(pt_jet)
        
        angular = {'deltaEta': deltaEta,
                   'deltaPhi': deltaPhi, 'deltaR': deltaR}
        energy = {'PT|PTjet': PT_PTjet, }
        return angular, energy

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, int]]:
        """The input shape is `(None, 8)`, where `None` indicates that the number of constituents is not fixed, 
        and `8` is the number of variables per constituent."""
        return (None, 3), (None, 1)



class InteractingRelativeConstituentVariables(ConstituentBase):
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

    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        m, pt, eta, phi, info_idx = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)

        E, px, py, pz = to_e_px_py_pz(m, pt, eta, phi)

        px_jet = tf.reduce_sum(px, axis=-1)
        py_jet = tf.reduce_sum(py, axis=-1)
        pz_jet = tf.reduce_sum(pz, axis=-1)
        
        pt_jet = tf.math.sqrt(px_jet**2 + py_jet**2)
        pt_jet = tf.clip_by_value(pt_jet, CLIP_MIN, CLIP_MAX)
        eta_jet = tf.math.asinh(pz_jet / pt_jet)
        phi_jet = tf.math.atan2(py_jet, px_jet)

        px = px / pt_jet
        py = py / pt_jet
        pz = pz / pt_jet
        
        # E_jet = tf.reduce_sum(E, axis=-1)
        # E = E / E_jet
        E = E / pt_jet
        pt = pt / tf.math.reduce_mean(pt_jet)
        eta = eta - tf.math.reduce_mean(eta_jet)
        phi = phi - tf.math.reduce_mean(phi_jet)

        delta, k_t, z, m2 = self.get_interacting_variables(m, pt, eta, phi, E, px, py, pz)

        interaction_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}

        deltaR = tf.math.sqrt(eta**2 + phi**2)

        logPT_PTjet = tf.math.log(tf.clip_by_value(pt, CLIP_MIN, CLIP_MAX))
        logE_Ejet = tf.math.log(tf.clip_by_value(E, CLIP_MIN, CLIP_MAX))
        
        m = tf.math.log(tf.clip_by_value(m, CLIP_MIN, CLIP_MAX))
        
        const_vars = {'log_PT|PTjet': logPT_PTjet, 'log_E|Ejet': logE_Ejet,
                        'deltaEta': eta, 'deltaPhi': phi, 'deltaR': deltaR, 'm': m}
        
        if len(self.const_types) > 2:
            for i, const_type in enumerate(self.const_types):
                const_vars[f'is_{const_type}'] = info_idx[:, i]
        elif len(self.const_types) == 2:
            const_vars[f'is_{self.const_types[1]}'] = info_idx
            
        return const_vars, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, None, int]]:
        """The input shape is `(None, 6)`, where `None` indicates that the number of constituents is not fixed, 
        and `6` is the number of variables per constituent."""
        if len(self.const_types) == 1:
            id_var = 0
        elif len(self.const_types) == 2:
            id_var = 1
        else:
            id_var = len(self.const_types) 
        return (None, 6+id_var), (None, None, 4)
    
class InteractingRelativeConstituentVariables2(ConstituentBase):
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

    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        m, pt, eta, phi, info_idx = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)

        E, px, py, pz = to_e_px_py_pz(m, pt, eta, phi)

        px_jet = tf.reduce_sum(px, axis=-1)
        py_jet = tf.reduce_sum(py, axis=-1)
        pz_jet = tf.reduce_sum(pz, axis=-1)
        
        pt_jet = tf.math.sqrt(px_jet**2 + py_jet**2) 
        pt_jet = tf.clip_by_value(pt_jet, CLIP_MIN, CLIP_MAX)
        eta_jet = tf.math.asinh(pz_jet / pt_jet)
        phi_jet = tf.math.atan2(py_jet, px_jet)
        
        E_jet = tf.reduce_sum(tf.clip_by_value(E, CLIP_MIN, CLIP_MAX), axis=-1)
        E_jet = tf.clip_by_value(E_jet, CLIP_MIN, CLIP_MAX)

        px = px / pt_jet
        py = py / pt_jet
        pz = pz / pt_jet
        
        E_un = E
        
        E = E / pt_jet
        pt = pt / pt_jet
        eta = eta - tf.math.reduce_mean(eta_jet)
        phi = phi - tf.math.reduce_mean(phi_jet)

        delta, k_t, z, m2 = self.get_interacting_variables(m, pt, eta, phi, E, px, py, pz)

        interaction_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}

        deltaR = tf.math.sqrt(eta**2 + phi**2)

        logPT_PTjet = tf.math.log(tf.clip_by_value(pt, CLIP_MIN, CLIP_MAX))
        logE_Ejet = tf.math.log(tf.clip_by_value(E_un / E_jet, CLIP_MIN, CLIP_MAX))
        
        # m = tf.math.log(tf.clip_by_value(m, CLIP_MIN, CLIP_MAX))
        
        const_vars = {'log_PT|PTjet': logPT_PTjet, 'log_E|Ejet': logE_Ejet,
                        'deltaEta': eta, 'deltaPhi': phi, 'deltaR': deltaR, 'm': m}
        
        if len(self.const_types) > 2:
            for i, const_type in enumerate(self.const_types):
                const_vars[f'is_{const_type}'] = info_idx[:, i]
        elif len(self.const_types) == 2:
            const_vars[f'is_{self.const_types[1]}'] = info_idx
            
        return const_vars, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, None, int]]:
        """The input shape is `(None, 6)`, where `None` indicates that the number of constituents is not fixed, 
        and `6` is the number of variables per constituent."""
        if len(self.const_types) == 1:
            id_var = 0
        elif len(self.const_types) == 2:
            id_var = 1
        else:
            id_var = len(self.const_types) 
        return (None, 6+id_var), (None, None, 4)

class InteractionConstituentVariables(ConstituentBase):
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
        m, pt, eta, phi, info_idx = self.get_constituents(sample)
        m_jet, pt_jet, eta_jet, phi_jet = self.get_jet(sample)
        E, px, py, pz = to_e_px_py_pz(m, pt, eta, phi)
        delta, k_t, z, m2 = self.get_interacting_variables(m, pt, eta, phi, E, px, py, pz)
        interaction_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}

        E = tf.math.sqrt(pt**2 + m**2)
        jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
        deltaEta = eta - tf.math.reduce_mean(eta_jet)
        deltaPhi = phi - tf.math.reduce_mean(phi_jet)
        deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

        logPT = tf.math.log(tf.clip_by_value(pt, CLIP_MIN, CLIP_MAX))

        logPT_PTjet = tf.math.log(tf.clip_by_value(pt / tf.math.reduce_sum(pt_jet), CLIP_MIN, CLIP_MAX))
        logE = tf.math.log(tf.clip_by_value(E, CLIP_MIN, CLIP_MAX))
        logE_Ejet = tf.math.log(tf.clip_by_value(E / tf.math.reduce_mean(jet_E), CLIP_MIN, CLIP_MAX))
        m = tf.math.log(tf.clip_by_value(m, CLIP_MIN, CLIP_MAX))
        const_vars = {'log_pT': logPT, 'log_PT|PTjet': logPT_PTjet, 'log_E': logE, 'log_E|Ejet': logE_Ejet,
                    'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR, 'm': m}
        
        if len(self.const_types) > 2:
            for i, const_type in enumerate(self.const_types):
                const_vars[f'is_{const_type}'] = info_idx[:, i]
        elif len(self.const_types) == 2:
            const_vars[f'is_{self.const_types[1]}'] = info_idx
        
        return const_vars, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[None, int], Tuple[None, None, int]]:
        """The input shape is a tuple of two tuples `(None, 8)` and `(None, None, 4)`, where the first tuple corresponds to the shape 
        of the variables for each constituent and the second tuple corresponds to the shape of a variable for each pair of constituents,
        i.e. a matrix for each jet."""
        if len(self.const_types) == 1:
            id_var = 0
        elif len(self.const_types) == 2:
            id_var = 1
        else:
            id_var = len(self.const_types) 
        return (None, 8 + id_var), (None, None, 4)

class JetClassVariables(TrainInput):
    
    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        
        kinematics = self.get_kinematics(sample)
        pid = self.get_pid(sample)
        trajectory_displacement = self.get_trajectory_displacement(sample)
        interaction_vars = self.get_interaction_variables(sample)
        
        return {**kinematics, **pid, **trajectory_displacement}, interaction_vars

    @property
    def input_shape(self) -> Tuple[Tuple[int, int], Tuple[int, int, int]]:
        return (self.max_constituents, 7+6+4), (self.max_constituents, self.max_constituents, 4)
        
    
    def get_kinematics(self, sample: ROOTVariables) -> Dict[str, tf.Tensor]:
        E_jet, pt_jet, eta_jet, phi_jet = sample['jet_energy'], sample['jet_pt'], sample['jet_eta'], sample['jet_phi']
        px, py, pz, E = sample['part_px'], sample['part_py'], sample['part_pz'], sample['part_energy']
        
        m, pt, eta, phi = to_m_pt_eta_phi(E, px, py, pz)
        
        delta_eta = eta - eta_jet
        delta_phi = cal_delta_phi(phi, phi_jet)
        delta_R = tf.math.sqrt(delta_eta**2 + delta_phi**2)
        log_pt = tf.math.log(tf.clip_by_value(pt, CLIP_MIN, CLIP_MAX))
        log_E = tf.math.log(tf.clip_by_value(E, CLIP_MIN, CLIP_MAX))
        log_pt_jet = tf.math.log(tf.clip_by_value(pt / pt_jet, CLIP_MIN, CLIP_MAX))
        log_E_jet = tf.math.log(tf.clip_by_value(E / E_jet, CLIP_MIN, CLIP_MAX))   
        
        return {'log_pT': log_pt, 'log_PT|PTjet': log_pt_jet, 'log_E': log_E, 'log_E|Ejet': log_E_jet,
                'deltaEta': delta_eta, 'deltaPhi': delta_phi, 'deltaR': delta_R}
    
    def get_pid(self, sample: ROOTVariables) -> Dict[str, tf.Tensor | tf.RaggedTensor]:
        charge = sample['part_charge']
        is_electron = sample['part_isElectron']
        is_muon = sample['part_isMuon']
        is_photon = sample['part_isPhoton']
        is_neutral_hadron = sample['part_isNeutralHadron']
        is_charged_hadron = sample['part_isChargedHadron']
        return {'charge': charge, 'is_electron': is_electron, 'is_muon': is_muon, 'is_photon': is_photon,
                'is_neutral_hadron': is_neutral_hadron, 'is_charged_hadron': is_charged_hadron}
        
    def get_trajectory_displacement(self, sample: ROOTVariables) -> Dict[str, tf.Tensor | tf.RaggedTensor]:
        d0_err = sample['part_d0err']
        dz_err = sample['part_dzerr']
        tanh_d0 = tf.tanh(sample['part_d0val'])
        tanh_dz = tf.tanh(sample['part_dzval'])
        return {'d0_err': d0_err, 'dz_err': dz_err, 'tanh_d0': tanh_d0, 'tanh_dz': tanh_dz}
    
    def get_interaction_variables(self, sample: ROOTVariables) -> Dict[str, tf.Tensor]:
        px, py, pz, E = sample['part_px'], sample['part_py'], sample['part_pz'], sample['part_energy']
        m, pt, y, phi = to_m_pt_y_phi(E, px, py, pz)
        
        delta_y = y[:, tf.newaxis] - y[tf.newaxis, :]
        delta_phi = cal_delta_phi(phi[:, tf.newaxis], phi[tf.newaxis, :])
        delta = tf.math.sqrt(delta_y**2 + delta_phi**2)
        
        k_t = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) * delta
        
        z = tf.math.minimum(pt[:, tf.newaxis], pt[tf.newaxis, :]) / \
            (pt[:, tf.newaxis] + pt[tf.newaxis, :])

        m2 = tf.math.square(E[:, tf.newaxis] + E[tf.newaxis, :]) - tf.math.square(px[:, tf.newaxis] + px[tf.newaxis, :]) - \
            tf.math.square(py[:, tf.newaxis] + py[tf.newaxis, :]) - \
            tf.math.square(pz[:, tf.newaxis] + pz[tf.newaxis, :])
            
        inter_vars = {'delta': delta, 'k_t': k_t, 'z': z, 'm2': m2}
            
        for key, value in inter_vars.items():
            inter_vars[key] = tf.linalg.set_diag(tf.math.log(tf.clip_by_value(value, CLIP_MIN, CLIP_MAX)), tf.zeros_like(m))
            
        return inter_vars
        
class JetClassVariablesNonInteraction(JetClassVariables):
    
    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        
        kinematics = self.get_kinematics(sample)
        pid = self.get_pid(sample)
        trajectory_displacement = self.get_trajectory_displacement(sample)
        
        return {**kinematics, **pid, **trajectory_displacement}

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (self.max_constituents, 7+6+4)

class JetClassVariablesPNet(JetClassVariables):
    
    def __call__(self, sample: ROOTVariables) -> Tuple[ROOTVariables, ROOTVariables]:
        kinematics = self.get_kinematics(sample)
        pid = self.get_pid(sample)
        trajectory_displacement = self.get_trajectory_displacement(sample)
        
        fts = {**kinematics, **pid, **trajectory_displacement}
        points = {'deltaEta': fts['deltaEta'], 'deltaPhi': fts['deltaPhi']}
        
        return fts, points

    @property
    def input_shape(self) -> Tuple[Tuple[int, int], Tuple[int, int, int]]:
        return (self.max_constituents, 7+6+4), (self.max_constituents, 2)
    
    
class JetClassHighlevel(TrainInput):
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
                'jet_tau4',
                'jet_eta',
                'jet_nparticles',
                'jet_tau3',
                'jet_sdmass',
                'jet_pt',
                'jet_tau2',
                'jet_phi',
                'jet_tau1',
                'jet_energy',]

    def __call__(self, sample: ROOTVariables) -> ROOTVariables:
        """Loops over the `per_jet_variables` and `per_event_variables` and constructs the input variables.

        Args:
            sample (ROOTVariables): The input sample.

        Returns:
            ROOTVariables: The output variables of the form `{'var_name': tf.Tensor}` where `var_name` is from `per_jet_variables` and `per_event_variables`.
        """

        new_sample = {var: tf.cast(sample[var], tf.float32)
                      for var in self.variables}
        return new_sample

    @property
    def input_shape(self) -> int:
        """The input shape is just an integer `len(self.variables)`."""
        return len(self.variables)
    
    
class InteractionConstituentVariablesTrack(InteractionConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_name = 'Track'

class ConstituentVariablesTopo(ConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower']
class RelativeConstituentVariablesTopo(RelativeConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower']

class InteractionConstituentVariablesTopo(InteractionConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower']

class InteractingRelativeConstituentVariablesTopo(InteractingRelativeConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower']

class InteractingRelativeConstituentVariablesTopo2(InteractingRelativeConstituentVariables2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower']

class IRCSVariablesTopo(IRCSVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower']
        
class ConstituentVariablesTopoTrack(ConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower', 'Track']

class InteractionConstituentVariablesTopoTrack(InteractionConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower', 'Track']

class InteractingRelativeConstituentVariablesTopoTrack(InteractingRelativeConstituentVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower', 'Track']

class IRCSVariablesTopoTrack(IRCSVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const_types = [self.const_name, 'TopoTower', 'Track']
        
        


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

    lookup_dict = {
                   'full_highlevel': FullHighLevelJetVariables,
                   'highlevel': HighLevelJetVariables,
                   'highlevel_no_eta': HighLevelJetVariablesNoEta,
                   'crafted_highlevel': CraftedHighLevelJetVariables,
                   'crafted_highlevel_no_eta': CraftedHighLevelJetVariablesNoEta,
                   'const': ConstituentVariables,
                   'rel_const': RelativeConstituentVariables,
                   'gnn': GNNVariables,
                   'int_const': InteractionConstituentVariables,
                   'int_rel_const': InteractingRelativeConstituentVariables,
                   'int_rel_const2': InteractingRelativeConstituentVariables2,
                   'irc_safe': IRCSVariables,
                   'int_const_track':InteractionConstituentVariablesTrack,
                   'const_topo': ConstituentVariablesTopo,
                   'rel_const_topo': RelativeConstituentVariablesTopo,
                   'int_const_topo': InteractionConstituentVariablesTopo,
                   'int_rel_const_topo': InteractingRelativeConstituentVariablesTopo,
                   'int_rel_const_topo2': InteractingRelativeConstituentVariablesTopo2,
                   'irc_safe_topo': IRCSVariablesTopo,
                   'const_topo_track': ConstituentVariablesTopoTrack,
                   'int_const_topo_track': InteractionConstituentVariablesTopoTrack,
                   'int_rel_const_topo_track': InteractingRelativeConstituentVariablesTopoTrack,
                   'irc_safe_topo_track': IRCSVariablesTopoTrack,
                   'jet_class': JetClassVariables,
                   'jet_class_non_int': JetClassVariablesNonInteraction,
                   'jet_class_highlevel': JetClassHighlevel,
                   'jet_class_pnet': JetClassVariablesPNet,
                   }

    if class_name not in lookup_dict.keys():
        raise ValueError(f'Unknown input class name {class_name}')

    return lookup_dict[class_name]


__pdoc__ = {f'{local_class.__name__}.__call__':
            True for local_class in TrainInput.__subclasses__() + [TrainInput]}
