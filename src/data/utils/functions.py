import tensorflow as tf
import vector
import awkward as ak

# def split_by_size(self, size: float) -> tuple[ROOTDataset, ROOTDataset]:
#     return ROOTDataset(self._dataset.take(int(size * self._dataset.cardinality().numpy())), self._variables), ROOTDataset(self._dataset.skip(int(size * self._dataset.cardinality().numpy())), self._variables)

def split_dataset(dataset: tf.data.Dataset, size: float) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    return dataset.take(int(size * dataset.cardinality().numpy())), dataset.skip(int(size * dataset.cardinality().numpy()))


def split_train_dev_test(dataset: tf.data.Dataset, test_size: float, dev_size: float) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_size = 1 - test_size - dev_size
    train, dev_test = split_dataset(dataset, train_size)
    dev, test = split_dataset(dev_test, dev_size / (1 - train_size))
    return train, dev, test


# def get_PFO_in_JetFrame(jets_pt: tf.Tensor,
#                         jets_eta: tf.Tensor,
#                         jets_phi: tf.Tensor,
#                         jets_m: tf.Tensor,
#                         jets_PFO_pt: tf.RaggedTensor,
#                         jets_PFO_eta: tf.RaggedTensor,
#                         jets_PFO_phi: tf.RaggedTensor,
#                         jets_PFO_m: tf.RaggedTensor):
#     vector.register_awkward()
#     Lvector = ak.Array({
#         "pt": jets_pt.numpy(),
#         "eta": jets_eta.numpy(),
#         "phi": jets_phi.numpy(),
#         "m": jets_m.numpy(),
#     }, with_name="Momentum4D")

#     awk = ak.Array({'pt': jets_PFO_pt.to_list(),
#                     'eta': jets_PFO_eta.to_list(),
#                     'phi': jets_PFO_phi.to_list(),
#                     'm': jets_PFO_m.to_list(),
#                     }, with_name="Momentum4D")
#     boosted_awk = awk.boost(-Lvector)

#     return [tf.RaggedTensor.from_nested_row_lengths(ak.to_list(ak.flatten(getattr(boosted_awk, var), axis=None)), nested_row_lengths=jets_PFO_pt.nested_row_lengths(), validate=True)
#             for var in ['pt', 'eta', 'phi', 'm']]


# @tf.function
# def py_func_get_PFO_in_JetFrame(sample: ROOTVariables) -> ROOTVariables:
#     sample = sample.copy()
#     input = [sample['jets_pt'], sample['jets_eta'], sample['jets_phi'], sample['jets_m'],
#              sample['jets_PFO_pt'], sample['jets_PFO_eta'], sample['jets_PFO_phi'], sample['jets_PFO_m']]
#     output = tf.py_function(get_PFO_in_JetFrame, inp=input, Tout=[
#                             tf.RaggedTensorSpec(shape=[None, None], dtype=tf.float32)]*4)
#     sample['jets_PFO_pt_JetFrame'], sample['jets_PFO_eta_JetFrame'], sample['jets_PFO_phi_JetFrame'], sample['jets_PFO_m_JetFrame'] = output
#     return sample
