import pytest 
from jidenn.data.jet_reclustering import create_jet, get_all_nodes, tree_to_tensor, tf_create_lund_plane
import tensorflow as tf
import numpy as np


def test_create_jet():
    px = [
        1.69886810e+02, 1.05292564e+02, 5.80032310e+01, 5.34577255e+01,
        2.24028816e+01, 1.46929998e+01, 1.40392914e+01, 1.27355070e+01,
        6.11943769e+00, 5.97744274e+00, 2.00730348e+00, 1.93206489e+00,
        1.02741814e+00, 2.73554802e-01, 1.18865050e-01
    ]
    py = [
        -177.77406, -74.58164, -64.93929, -33.387066, -23.627367,
        -14.329296, -13.579385, -9.271716, -6.4544373, -3.3738763,
        -2.3707838, -1.4003158, -1.0385578, -0.66247356, -0.6830277
    ]
    pz = [
        567.37463, 417.92645, 203.2926, 203.11565, 76.22963, 60.205322,
        46.17678, 51.58491, 20.227394, 17.294386, 9.763338, 5.8184795,
        3.8036668, 2.9332755, 2.7955122
    ]
    E = [
        618.36835, 437.39194, 221.15463, 212.66968, 82.8922, 63.607338,
        50.137775, 53.93674, 22.096928, 18.606684, 10.2575035, 6.290305,
        4.074564, 3.0227947, 2.8835783
    ]
    

    tensor, node_4mom = tf_create_lund_plane(tf.constant(px), tf.constant(py), tf.constant(pz), tf.constant(E))
    # set max rows to print to None  
    # set display width to 0
    
    np.set_printoptions(linewidth=100, threshold=np.inf)
    print(tensor.numpy())
    print(node_4mom.numpy())