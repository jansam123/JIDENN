import tensorflow as tf
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

# maximum number of threads we allow to utilize, 0 for all
MAX_THREADS = 12
# seed for reproducibility
SEED = 42
# batch size
BATCH_SIZE = 64
# num of batches (jets//BATCH_SIZE) used for testing
TEST_SIZE = 50
# num of batches (jets//BATCH_SIZE) used to calculate mean and variance for normalization
NORMALIZATION_SIZE = 10
# hidden layers, with given sizes, used in a model
HIDDEN_LAYERS = [128, 128]
# probability that we will mask a node in hidden layer to avoid overfitting
DROPOUT = 0.1
# number of epochs we want to train for, ie. number of times we show the model our entire dataset
EPOCHS = 10

# name of the TTree in a root file we want to load
TTREE = 'NOMINAL'
# root files that are going to be utilized
# in CHIMERA use:
base_path = '/work/ucjf-atlas/plesv6am/for_sam/qg/jetProp4/jetProp4/user.pleskot.mc16_13TeV.364704.JETM13.e7142_s3126_r10724_p4277.jetProp4_ANALYSIS'
if not os.path.exists(base_path):
    base_path = '/home/jankovys/JIDENN/data/pythia_exmaples'
# in GPULAB use:
FILENAMES = [f'{base_path}/user.pleskot.31142736.ANALYSIS._000002.root']
# f'{base_path}/user.pleskot.31142736.ANALYSIS._000001.root',
# f'{base_path}/user.pleskot.31142736.ANALYSIS._000003.root',
# f'{base_path}/user.pleskot.31142736.ANALYSIS._000004.root', ]

# name of a variable that is going to be our label, ie. what we want to predict
LABEL = 'jets_PartonTruthLabelID'
# this labels are used for mapping from traditional variables to 0 (gluons) and 1 (quarks) in our case
# if we want to tag something different (eg. b-Tagging) just arrage the quarks and gluons accordingly
FIRST_LABEL = [21]
SECOND_LABEL = [1, 2, 3, 4, 5, 6]
# variables which are going to be used in training
# some variables contain slicing (ie. jets_ChargedPFOWidthPt1000[:,:,0]), this just picks the first value per jet
# first index is for events, second for jets and third for a given variable per Jet
VARIABLES = ['jets_ActiveArea4vec_eta', 'jets_ActiveArea4vec_m', 'jets_ActiveArea4vec_phi', 'jets_ActiveArea4vec_pt', 'jets_ChargedPFOWidthPt1000[:,:,0]',
             'jets_DetectorEta', 'jets_EMFrac', 'jets_EnergyPerSampling[:,:,0]', 'jets_FracSamplingMax', 'jets_FracSamplingMaxIndex', 'jets_GhostMuonSegmentCount',
             'jets_JetConstitScaleMomentum_eta', 'jets_JetConstitScaleMomentum_m', 'jets_JetConstitScaleMomentum_phi', 'jets_JetConstitScaleMomentum_pt',
             'jets_JVFCorr', 'jets_Jvt', 'jets_JvtRpt', 'jets_NumChargedPFOPt1000[:,:,0]', 'jets_NumChargedPFOPt500[:,:,0]', 'jets_NumTrkPt1000[:,:,0]',
             'jets_NumTrkPt500[:,:,0]', 'jets_SumPtChargedPFOPt500[:,:,0]', 'jets_SumPtTrkPt500[:,:,0]', 'jets_Timing', 'jets_TrackWidthPt1000[:,:,0]', 'jets_Width',
             'jets_chf', 'jets_eta', 'jets_fJVT', 'jets_m', 'jets_passFJVT', 'jets_passJVT', 'jets_phi', 'jets_pt']

# this is just so that you can override the default values in a terminal with an appropriate flags
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=BATCH_SIZE, type=int, help="Batch size.")
parser.add_argument("--epochs", default=EPOCHS, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=SEED, type=int, help="Random seed.")
parser.add_argument("--threads", default=MAX_THREADS, type=int, help="Maximum number of threads to use.")
parser.add_argument("--test_size", default=TEST_SIZE, type=int, help="Num of jets//BATCH_SIZE used for testing.")
parser.add_argument("--norm_size", default=NORMALIZATION_SIZE, type=int,
                    help="Num of jets//BATCH_SIZE used to calculate the mean and variance of variables.")
parser.add_argument("--hidden_layers", default=HIDDEN_LAYERS, nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--dropout", default=DROPOUT, type=float, help="Size of dropout after each layer.")
parser.add_argument("--ttree", default=TTREE, type=str, help="Name of a TTree in root files.")
parser.add_argument("--root_files", default=FILENAMES, type=str, nargs="*", help="ROOT files.")
parser.add_argument("--label", default=LABEL, type=str, help="Name of the label in the TTree we want to predict.")
parser.add_argument("--variables", default=VARIABLES, type=str, nargs="*",
                    help="Name of jet variables in the TTree used in making prediction.")
parser.add_argument("--first_label", default=FIRST_LABEL, nargs="*",
                    type=int, help="First label, we want to predict (gluon).")
parser.add_argument("--second_label", default=SECOND_LABEL, nargs="*",
                    type=int, help="Second label, we want to predict (quarks).")


def main(args: argparse.Namespace) -> None:
    # restrict the maximum number of available threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    # set a seed for utilizing random variables
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # The simplest way to load root files is to dump them in a pandas dataframe.
    # In this example it is done sequantially (not the best way, but easy to understand).
    # create empty dataset to which we will appned
    df = pd.DataFrame()
    print("Loading ROOT files...(this may take a minute)")
    for file in args.root_files:
        print(f"Loading {file}")
        # open a root file and pick TTree
        tree = uproot.open(file, num_workers=args.threads)[args.ttree]
        # load all variables and labels in a awkward array
        # by setting library='pd' we can load directly to pd.DataFrame, but there is bug when TBranches are nested NTuples
        arr = tree.arrays([*args.variables, args.label], library='ak')
        # convert awkward array to pandas dataframe
        arr = ak.to_pandas(arr)
        # concat to big dataframe
        df = pd.concat([df, arr])

    print(df.head())
    # droping index converts rows indexed by events and jets to be indexed only by jets
    df = df.reset_index(drop=True)
    # cast all variables to float, so they can be in one array (tensor)
    df[args.variables] = df[args.variables].astype(np.float32)
    # we use this pandas query magic, to split the dataset into a gluon one and quark one
    # the '@' sign is used when refering to external variables (column names don't need it)
    gluons = df.query(f'{args.label} in @args.first_label')
    quarks = df.query(f'{args.label} in @args.second_label')
    # convert pandas dataframes to tf datasets
    # tf.data.Dataset can't convert pd.DataFrame directly, that is way we convert it to dict first
    quark_dataset = tf.data.Dataset.from_tensor_slices(dict(quarks))
    gluon_dataset = tf.data.Dataset.from_tensor_slices(dict(gluons))
    # from now on, all the transformations on tf.data.Dataset will be just a set of instructions and won't be executed until training
    # combine the two datasets with equal probability to obtain a balanced dataset
    mixed_dataset = tf.data.Dataset.sample_from_datasets([quark_dataset, gluon_dataset], weights=[
        0.5, 0.5], stop_on_empty_dataset=True)

    @tf.function
    def prepare_sample(sample: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        # roughly speeking, a sample from mixed dataset will be a dict of floats and one int (label)
        # this function converts the dictionary of variables to a tensor and an int (label)
        # tf.function decorator is another black magic of tensorflow, which allows for faster computation in a Graph Mode
        label = sample[args.label]
        # map original lables [1,2,3,4,5,6,21] into just quarks and just gluons [1,0]
        label = tf.cast(tf.reduce_any(tf.equal(label, tf.constant(args.second_label))), tf.int32)
        # convert individual variables to one tensor
        # it shouldn't really hurt to iterate here the variables for every sample, becouse it is optimized by tf in Graph Mode (black magic)
        data = tf.stack([sample[var] for var in args.variables])
        return data, label

    # use the function to map every element
    mixed_dataset = mixed_dataset.map(prepare_sample)
    # group samples to batches
    mixed_dataset = mixed_dataset.batch(args.batch_size)
    # prefetching intructs the dataset to utilize more cores for data preparation, so the DNN doesn't wait for the data
    # you can specify a concrete number of cores or pass the tf.data.AUTOTUNE, which will do it dynamically
    mixed_dataset = mixed_dataset.prefetch(tf.data.AUTOTUNE)

    # split the dataset to train part and test part
    test_dataset = mixed_dataset.take(args.test_size)
    train_dataset = mixed_dataset.skip(args.test_size)

    # we need to create the normalization layer before other layers, because we need to calculate the variance and mean of each variable
    normalizer = tf.keras.layers.Normalization()

    @ tf.function
    def pick_only_data(data: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        # this function is just a helper to choose only data and not label
        # we could use a lambda syntax, but sometimes tf has a problem converting it to a Graph Mode
        return data
    # adapt method will calculate the means and variances, and set them as weights of the layer
    # again we use the mapping to pick just data with predefined function
    # we can limit the number of steps, to make it faster (steps=jets//BATCH_SIZE)
    normalizer.adapt(train_dataset.map(pick_only_data), steps=args.norm_size)

    # only here we start to build the network
    # we utilize the so called 'functional API' (https://www.tensorflow.org/guide/keras/functional)
    # input layer needs to know the shape of an sample, here we omit the batch dimension
    input = tf.keras.layers.Input(shape=(len(args.variables), ))
    # firstly we apply the normalization layer on the input
    hidden = normalizer(input)
    # using the fucntional API it is easy to create multiple hidden layers in a loop
    for layer_size in args.hidden_layers:
        # we create a Dense layer (ie. Liner Perceptron, ie. Fully Connected Layer) with the ReLU = max(0, x) activation function
        hidden = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu)(hidden)
        # after each dense layer it is 99% of the time good to use the dropout layer to avoid overfitting
        # this layer masks (drops out) a node from previous layer with probabilty 'DROPOUT', this is done for all nodes from previous layer
        # 90% of the time it is good to set the DROPOUT to 0.5 and rather increase the number of layer_size
        # in our case, where we have a lot of data that are randomly sampled overfitting is not an issue, so it is not neccesery
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
    # output layer is just one number between 0 and 1, thats why we use the sigmoid=1/(1+exp(-x)) activation to get it
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)

    # TODO:
    # try to change the basic FC layers to highway networks (https://arxiv.org/abs/1505.00387)
    # it is calculated using the following formula: y = H(x) * T(x) + x * (1 - T(x))
    # where H(x) is the output of the FC layer, T(x) is the output of the sigmoid layer and x is the input
    # you can either add (multiply) two layers or use the tf.keras.layers.Add() (tf.keras.layers.Multiply()) layer
    # be careful when adding two layers, because it expects the same shape of the inputs
    # END TODO

    # here we create the model with input and output
    model = tf.keras.Model(inputs=input, outputs=output)
    # we need to compile the model, ie. to specify the loss we want to minimize,
    # optimizer (which is used to minimize the loss) and metrics (which are used to evaluate the model)
    # in our case we use the binary crossentropy loss, which is good for binary classification (ie. quark or gluon)
    # we use the Adam optimizer, which is a good default choice
    # we use the accuracy metric, which is just the fraction of correct predictions and AUC, which is the area under the ROC curve
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    # here we print the model summary, outline
    model.summary()
    # this is the part where the training happens, we specify the number of epochs we want to train for and validation dataset
    # validation will happen after each epoch
    model.fit(train_dataset, epochs=args.epochs, validation_data=test_dataset)
    # this is just a demonstration of evalution the model
    model.evaluate(test_dataset)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
