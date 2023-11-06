from typing import List
import tensorflow as tf
import numpy as np


import jidenn.config.augmentation_config as aug_cfg
import jidenn.config.config as cfg


def get_drop_constituents(config: aug_cfg.DropSoft):
    min_const = config.min_number_consti
    skew = config.skew
    center_frac = config.center_location

    @tf.function
    def sigmoid(x, a=1, b=0):
        return - 1. / (1. + tf.exp(a * (x - b))) + 1.

    @tf.function
    def drop_constituents(jets) -> tf.Tensor:
        jets = jets.copy()
        PFO_pt = jets['jets_PFO_pt']
        n_consts = tf.shape(PFO_pt)[0]
        if n_consts < min_const:
            return tf.range(0, n_consts, 1, dtype=tf.int32)

        center = tf.cast((tf.cast(n_consts, tf.float32) * center_frac), tf.float32)

        coin_flips = tf.random.uniform(shape=(n_consts,))
        idxs = tf.range(0, n_consts, 1)
        drop_probabilty = sigmoid(tf.cast(idxs, tf.float32), b=center, a=skew)

        take_index = tf.where(coin_flips > drop_probabilty, True, False)
        if tf.reduce_all(take_index == False):
            return tf.range(0, n_consts, 1, dtype=tf.int32)

        jets['jets_PFO_m'] = jets['jets_PFO_m'][take_index]
        jets['jets_PFO_pt'] = jets['jets_PFO_pt'][take_index]
        jets['jets_PFO_eta'] = jets['jets_PFO_eta'][take_index]
        jets['jets_PFO_phi'] = jets['jets_PFO_phi'][take_index]
        return jets
    return drop_constituents


def get_rotation_augm(config: aug_cfg.Rotation):
    max_angle = config.max_angle
    max_angle *= np.pi / 180

    @tf.function
    def random_rotate_around_fixed_axis(vector, axis):
        axis = axis / tf.norm(axis)
        theta = tf.random.uniform(shape=[], minval=0, maxval=max_angle)
        a = tf.cos(theta) * vector
        b = tf.sin(theta) * (tf.linalg.cross(tf.broadcast_to(axis, shape=tf.shape(vector)), vector))
        c = tf.einsum('i,ji->j', axis, vector)
        d = (1 - tf.cos(theta)) * axis
        return a + b + tf.einsum('j,i->ji', c, d)

    @tf.function
    def rotation_augm(jets):
        jets = jets.copy()
        jets = add_cartesian_coordinates(jets)
        axis = tf.stack([jets['jets_px'], jets['jets_py'], jets['jets_pz']])
        vector = tf.stack([jets['jets_PFO_px'], jets['jets_PFO_py'], jets['jets_PFO_pz']], axis=1)
        rotated_vector = random_rotate_around_fixed_axis(vector, axis)
        jets['jets_PFO_px'], jets['jets_PFO_py'], jets['jets_PFO_pz'] = tf.unstack(rotated_vector, axis=1)
        jets['jets_PFO_m'], jets['jets_PFO_pt'], jets['jets_PFO_eta'], jets['jets_PFO_phi'] = to_m_pt_eta_phi(
            jets['jets_PFO_E'], jets['jets_PFO_px'], jets['jets_PFO_py'], jets['jets_PFO_pz'])
        return jets
    return rotation_augm


def get_boost_augm(config: aug_cfg.Boost):
    max_beta = config.max_beta

    @tf.function
    def random_boost(energy, momentum):
        n = tf.random.normal(shape=[3])
        n = n / tf.norm(n)
        beta = n * tf.random.uniform(shape=[], minval=0, maxval=max_beta)
        gamma = 1 / tf.sqrt(1 - tf.norm(beta)**2)
        beta_p = tf.einsum('i,ji->j', beta, momentum)
        new_energy = gamma * (energy - beta_p)
        new_momentum = momentum + tf.einsum('i,j->ji', beta, (-gamma * energy +
                                            (gamma - 1) * beta_p / tf.norm(beta)**2))
        return new_energy, new_momentum

    @tf.function
    def boost_augm(jets):
        jets = jets.copy()
        jets = add_cartesian_coordinates(jets)
        momentum = tf.stack([jets['jets_PFO_px'], jets['jets_PFO_py'], jets['jets_PFO_pz']], axis=1)
        jets['jets_PFO_E'], new_momentum = random_boost(jets['jets_PFO_E'], momentum)
        jets['jets_PFO_px'], jets['jets_PFO_py'], jets['jets_PFO_pz'] = tf.unstack(new_momentum, axis=1)
        jets['jets_PFO_m'], jets['jets_PFO_pt'], jets['jets_PFO_eta'], jets['jets_PFO_phi'] = to_m_pt_eta_phi(
            jets['jets_PFO_E'], jets['jets_PFO_px'], jets['jets_PFO_py'], jets['jets_PFO_pz'])
        return jets
    return boost_augm


def get_random_split_fn(config: aug_cfg.CollinearSplit):
    prob = config.splitting_amount

    @tf.function
    def random_split(jets):
        jets = jets.copy()
        m, pt, eta, phi = jets['jets_PFO_m'], jets['jets_PFO_pt'], jets['jets_PFO_eta'], jets['jets_PFO_phi']
        split_index = tf.where(tf.random.uniform(shape=tf.shape(pt), minval=0, maxval=1) < prob, True, False)

        frac = tf.random.uniform(shape=tf.shape(pt), minval=0., maxval=1.)
        new_all_pt = pt * tf.cast(tf.logical_not(split_index), tf.float32) + \
            pt * tf.cast(split_index, tf.float32) * frac
        new_pt = tf.boolean_mask(pt, split_index) * (1 - tf.boolean_mask(frac, split_index))
        new_m = tf.zeros_like(new_pt)
        new_eta = tf.boolean_mask(eta, split_index)
        new_phi = tf.boolean_mask(phi, split_index)
        jets['jets_PFO_m'] = tf.concat([m, new_m], axis=0)
        jets['jets_PFO_pt'] = tf.concat([new_all_pt, new_pt], axis=0)
        jets['jets_PFO_eta'] = tf.concat([eta, new_eta], axis=0)
        jets['jets_PFO_phi'] = tf.concat([phi, new_phi], axis=0)
        return jets
    return random_split


def get_soft_smear_fn(config: aug_cfg.SoftSmear):
    scale = config.energy_scale

    @tf.function
    def get_soft_smear(jets):
        jets = jets.copy()
        jets['jets_PFO_eta'] = tf.random.normal(shape=tf.shape(
            jets['jets_PFO_eta']), mean=jets['jets_PFO_eta'], stddev=scale / jets['jets_PFO_pt'])
        jets['jets_PFO_phi'] = tf.random.normal(shape=tf.shape(
            jets['jets_PFO_phi']), mean=jets['jets_PFO_phi'], stddev=scale / jets['jets_PFO_pt'])
        return jets
    return get_soft_smear


aug_name_mapping = {
    'drop_soft': get_drop_constituents,
    'rotation': get_rotation_augm,
    'boost': get_boost_augm,
    'collinear_split': get_random_split_fn,
    'soft_smear': get_soft_smear_fn,
}


def construct_augmentation(cfgs_aug: cfg.Augmentations):
    for aug_name in cfgs_aug.order:
        if aug_name not in aug_name_mapping:
            raise ValueError(f"Augmentation {aug_name} not implemented, use one of {list(aug_name_mapping.keys())}")
        

    @tf.function
    def augmentation(jets):
        for aug_name in cfgs_aug.order:
            if tf.random.uniform(shape=()) < getattr(cfgs_aug, aug_name).prob:
                jets = aug_name_mapping[aug_name](getattr(cfgs_aug, aug_name))(jets)
        return jets

    return augmentation
