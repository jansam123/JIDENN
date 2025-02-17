import fastjet as fj
import tensorflow as tf
import awkward as ak
import numpy as np


class JetTree:
    """JetTree keeps track of the tree structure of a jet declustering."""

    ktmin = 0.0
    deltamin = 0.0

    # ----------------------------------------------------------------------
    def __init__(self, pseudojet, child=None):
        """Initialize a new node, and create its two parents if they exist."""
        self.harder = None
        self.softer = None
        self.delta2 = 0.0
        self.lundCoord = None
        # if it has a direct child (i.e. one level further up in the
        # tree), give a link to the corresponding tree object here
        self.child = child

        while True:
            j1 = fj.PseudoJet()
            j2 = fj.PseudoJet()
            if pseudojet and pseudojet.has_parents(j1, j2):
                # order the parents in pt
                if (j2.pt() > j1.pt()):
                    j1, j2 = j2, j1
                # check if we satisfy cuts
                delta = j1.delta_R(j2)
                kt = j2.pt() * delta
                if (delta < JetTree.deltamin):
                    break
                # then create two new tree nodes with j1 and j2
                if kt >= JetTree.ktmin:
                    self.harder = JetTree(j1, child=self)
                    self.softer = JetTree(j2, child=self)
                    break
                else:
                    pseudojet = j1
            else:
                break

        # finally define the current node
        self.node = np.array([pseudojet.px(), pseudojet.py(), pseudojet.pz(), pseudojet.E()], dtype='float32')

def create_jet(px, py, pz, E):
    jet_def = fj.JetDefinition(fj.cambridge_algorithm, 1000.)
    pseudojets = [fj.PseudoJet(px[i], py[i], pz[i], E[i]) for i in range(len(px))]
    jet = jet_def(pseudojets)[0]
    return jet

def get_all_nodes(jet_tree):
    print(jet_tree.node)
    if jet_tree.softer is None and jet_tree.harder is not None:
        return  get_all_nodes(jet_tree.harder)
    elif jet_tree.harder is None and jet_tree.softer is not None:
        return get_all_nodes(jet_tree.softer)
    elif jet_tree.softer is None and jet_tree.harder is None:
        return
    else:
        return get_all_nodes(jet_tree.harder), get_all_nodes(jet_tree.softer)
    
    
    
def flatten_jet_tree(root):
    """
    Flatten the JetTree into a list of nodes using a depth-first traversal.
    """
    nodes = []
    def traverse(node):
        if node is None:
            return
        nodes.append(node)
        traverse(node.harder)
        traverse(node.softer)
    traverse(root)
    return nodes

def tree_to_tensor(root):
    """
    Convert the JetTree graph to a TensorFlow tensor of shape [M, M, 4],
    where M is the number of nodes and each [i,j] element holds a 4-vector.
    
    The convention here is:
      - The diagonal [i,i] holds the 4-momentum of node i.
      - For each node i, if it has a direct child node j (from either the
        'harder' or 'softer' branch), then tensor[i,j] holds the child’s 4-momentum.
      - All other entries are zeros.
    """
    # 1. Traverse the tree and collect nodes.
    nodes = flatten_jet_tree(root)
    node_4mom = np.stack([node.node for node in nodes])
    
    M = len(nodes)
    
    # 2. Create an empty tensor (NumPy array) of shape (M, M, 4)
    data = np.zeros((M, M), dtype=np.float32)
    
    # 3. Create a mapping from node object to index.
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    
    # 4. Fill the tensor:
    for idx, node in enumerate(nodes):
        # For each direct child (harder and softer), record the link:
        if node.harder is not None:
            jdx = node_to_index[node.harder]
            data[idx, jdx] = 1. #node.harder.node
        if node.softer is not None:
            jdx = node_to_index[node.softer]
            data[idx, jdx] = 1. #node.softer.node

    # Convert the numpy array to a TensorFlow tensor.
    tensor = tf.convert_to_tensor(data)
    node_4mom = tf.convert_to_tensor(node_4mom)
    return tensor, node_4mom

def tf_create_lund_plane(px: tf.Tensor, py:tf.Tensor, pz: tf.Tensor, E: tf.Tensor):
    jet = create_jet(px=px.numpy().tolist(), py=py.numpy().tolist(), pz=pz.numpy().tolist(), E=E.numpy().tolist())
    
    jet_tree = JetTree(jet)
    lund_graph, node_4mom = tree_to_tensor(jet_tree)
    return lund_graph, node_4mom