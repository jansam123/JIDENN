"""
This module contains functions to transform 4-vectors between different representations.
It also contains a functions for various 4-vector operations such as boosts and rotations.
"""
from typing import Tuple
import tensorflow as tf
import numpy as np


@tf.function
def to_e_px_py_pz(mass, pt, eta, phi) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    px = pt * tf.cos(phi)
    py = pt * tf.sin(phi)
    pz = pt * tf.sinh(eta)
    E = tf.sqrt(mass**2 + px**2 + py**2 + pz**2)
    # E = pt * tf.cosh(eta)
    return E, px, py, pz


@tf.function
def to_m_pt_eta_phi(E: tf.Tensor, px: tf.Tensor, py: tf.Tensor, pz: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Converts a 4-vector in the form (e, px, py, pz) to (mass, pt, eta, phi)

    Args:
        E (tf.Tensor): tensor of shape (N,) with energy
        px (tf.Tensor): tensor of shape (N,) with px
        py (tf.Tensor): tensor of shape (N,) with py
        pz (tf.Tensor): tensor of shape (N,) with pz

    Returns:
        tf.Tensor: tensor of shape (N, 4) with (mass, pt, eta, phi)
    """
    pt = tf.sqrt(px**2 + py**2)
    p = tf.sqrt(pt**2 + pz**2)
    eta = tf.math.atanh(pz / p)
    phi = tf.math.atan2(py, px)
    mass = tf.sqrt(E**2 - pt**2 - pz**2)
    return mass, pt, eta, phi


@tf.function
def to_m_pt_y_phi(E: tf.Tensor, px: tf.Tensor, py: tf.Tensor, pz: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Converts a 4-vector in the form (e, px, py, pz) to (mass, pt, y, phi)

    Args:
        E (tf.Tensor): tensor of shape (N,) with energy
        px (tf.Tensor): tensor of shape (N,) with px
        py (tf.Tensor): tensor of shape (N,) with py
        pz (tf.Tensor): tensor of shape (N,) with pz

    Returns:
        tf.Tensor: tensor of shape (N, 4) with (mass, pt, eta, phi)
    """
    pt = tf.sqrt(px**2 + py**2)
    p = tf.sqrt(pt**2 + pz**2)
    y = 0.5 * tf.math.log((E + pz) / (E - pz))
    phi = tf.math.atan2(py, px)
    mass = tf.sqrt(E**2 - pt**2 - pz**2)
    return mass, pt, y, phi


@tf.function
def cal_delta_phi(phi1: tf.Tensor, phi2: tf.Tensor) -> tf.Tensor:
    """Calculates the difference in phi between two 4-vectors"""
    
    return tf.math.atan2(tf.math.sin(phi1 - phi2), tf.math.cos(phi1 - phi2))


