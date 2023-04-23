"""
This module contains functions to transform 4-vectors between different representations.
It also contains a functions for various 4-vector operations such as boosts and rotations.
"""

import tensorflow as tf
import numpy as np


@tf.function
def to_e_px_py_pz(x: tf.Tensor) -> tf.Tensor:
    """Converts a 4-vector in the form (mass, pt, eta, phi) to (e, px, py, pz)

    Args:
        x (tf.Tensor): tensor of shape (N, 4) with (mass, pt, eta, phi)

    Returns:
        tf.Tensor: tensor of shape (N, 4) with (E, px, py, pz)
    """
    mass, pt, eta, phi = tf.unstack(x, axis=1)
    px = pt * tf.cos(phi)
    py = pt * tf.sin(phi)
    pz = pt * tf.tanh(eta)
    E = tf.sqrt(mass**2 + px**2 + py**2 + pz**2)
    return tf.stack([E, px, py, pz], axis=1)


@tf.function
def to_m_pt_eta_phi(x: tf.Tensor) -> tf.Tensor:
    """Converts a 4-vector in the form (e, px, py, pz) to (mass, pt, eta, phi)

    Args:
        x (tf.Tensor): tensor of shape (N, 4) with (E, px, py, pz)

    Returns:
        tf.Tensor: tensor of shape (N, 4) with (mass, pt, eta, phi)
    """
    E, px, py, pz = tf.unstack(x, axis=1)
    pt = tf.sqrt(px**2 + py**2)
    eta = tf.math.atanh(pz/pt)
    phi = tf.math.atan2(py, px)
    mass = tf.sqrt(E**2 - pt**2 - pz**2)
    return tf.stack([mass, pt, eta, phi], axis=1)


@tf.function
def generate_boost_matrix(phi: float, n: tf.Tensor) -> tf.Tensor:
    """Generates a boost matrix for a given rapidity and directionn n

    Args:       
        phi (float): rapidity of the boost
        n (tf.Tensor): Tensor of shape (3,) with the direction of the boost

    Returns:
        tf.Tensor: boost matrix
    """
    B = [[tf.cosh(phi), -n[0]*tf.sinh(phi), -n[1]*tf.sinh(phi), -n[2]*tf.sinh(phi)],
         [-n[0]*tf.sinh(phi), 1 + (n[0]**2)*(tf.cosh(phi)-1), n[0]*n[1]*(tf.cosh(phi)-1), n[0]*n[2]*(tf.cosh(phi)-1)],
         [-n[1]*tf.sinh(phi), n[1]*n[0]*(tf.cosh(phi)-1), 1 + (n[1]**2)*(tf.cosh(phi)-1), n[1]*n[2]*(tf.cosh(phi)-1)],
         [-n[2]*tf.sinh(phi), n[2]*n[0]*(tf.cosh(phi)-1), n[2]*n[1]*(tf.cosh(phi)-1), 1 + (n[2]**2)*(tf.cosh(phi)-1)]]
    return tf.convert_to_tensor(B, dtype=tf.float32)


@tf.function
def boost(x: tf.Tensor, boost_matrix: tf.Tensor) -> tf.Tensor:
    """Boosts a 4-vector x by a boost matrix

    Args:
        x (tf.Tensor): tensor of shape (N, 4) with (E, px, py, pz)
        boost_matrix (tf.Tensor): boost matrix

    Returns:
        tf.Tensor: boosted tensor of shape (N, 4) with (E, px, py, pz)
    """
    return tf.einsum('ij,aj->ai', boost_matrix, x)


@tf.function
def rotate(x: tf.Tensor, rotation_matrix: tf.Tensor) -> tf.Tensor:
    """Rotates multiple 4-vectors (E,px,py,pz) x by rotation_matrix

    Args:
        x (tf.Tensor): Tensor of shape (N, 4), where N is the number of 4-vectors of shape (E,px,py,pz) to be rotated
        rotation_matrix (tf.Tensor): Tensor of shape (3, 3) containing the rotation matrix

    Returns:
        tf.Tensor: Tensor of shape (N, 4) containing the rotated 4-vectors
    """
    return tf.concat([x[:, 0:1], tf.einsum("ij,jk->ik", x[:, 1:], rotation_matrix)], axis=1)


@tf.function
def generate_rotation_matrix(theta: float, n: tf.Tensor) -> tf.Tensor:
    """Generates a rotation matrix for a rotation around an arbitrary axis

    Args:
        theta (float): The angle to rotate by
        n (tf.Tensor): The axis to rotate around of shape (3,)

    Returns:
        tf.Tensor: The rotation matrix of shape (3, 3)
    """
    R = [[tf.cos(theta) + n[0] ** 2 * (1 - tf.cos(theta)), n[0] * n[1] * (1 - tf.cos(theta)) - n[2] * tf.sin(theta), n[0] * n[2] * (1 - tf.cos(theta)) + n[1] * tf.sin(theta)],
         [n[1] * n[0] * (1 - tf.cos(theta)) + n[2] * tf.sin(theta), tf.cos(theta) + n[1] ** 2 *
          (1 - tf.cos(theta)), n[1] * n[2] * (1 - tf.cos(theta)) - n[0] * tf.sin(theta)],
         [n[2] * n[0] * (1 - tf.cos(theta)) - n[1] * tf.sin(theta), n[2] * n[1] * (1 - tf.cos(theta)) + n[0] * tf.sin(theta), tf.cos(theta) + n[2] ** 2 * (1 - tf.cos(theta))]]
    R = tf.convert_to_tensor(R)
    return R


@tf.function
def random_lorentz_transform(x: tf.Tensor) -> tf.Tensor:
    """Randomly applies a lorentz transform to multiple 4-vectors (E,px,py,pz) x

    Args:
        x (tf.Tensor): Tensor of shape (N, 4), where N is the number of 4-vectors of shape (E,px,py,pz) to be transformed

    Returns:
        tf.Tensor: Tensor of shape (N, 4) containing the transformed 4-vectors
    """
    # Randomly generate rapidity phi
    phi = tf.random.uniform(shape=(), minval=-1, maxval=1, dtype=tf.float32)
    phi = tf.math.atanh(phi)
    # Randomly generate direction n_b
    n_b = tf.random.normal(shape=(3,), dtype=tf.float32)
    n_b = n_b/tf.norm(n_b)
    # Randomly generate theta
    theta = tf.random.uniform(shape=(), minval=0, maxval=2 * np.pi)
    # Randomly generate n for rotation
    n_r = tf.random.uniform(shape=(3,), minval=-1, maxval=1)
    n_r = n_r / tf.norm(n_r)
    # Generate rotation matrix
    R = generate_rotation_matrix(theta, n_r)
    #
    B = generate_boost_matrix(phi, n_b)
    # Apply rotation
    # x = rotate(x, R)
    # Apply boost
    x = boost(x, B)
    return x


@tf.function
def lorentz_dot_product(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Calculate the lorentz dot product (+,-,-,-) of two 4-vectors 

    Args:
        x (tf.Tensor): Tensor of shape (N, 4) containing the first 4-vectors
        y (tf.Tensor): Tensor of shape (N, 4) containing the second 4-vectors

    Returns:
        tf.Tensor: Tensor of shape (N, N) containing the lorentz dot product of 
    """
    metric = tf.convert_to_tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=tf.float32)
    return tf.einsum('nj,jk,mk->nm', x, metric, y)


