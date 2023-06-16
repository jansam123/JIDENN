import tensorflow_datasets as tfds
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pt_dist(dataset: tf.data.Dataset, save_path: str = 'figs.png') -> None:

    @tf.function
    def get_labeled_pt(data):
        parton = data['jets_PartonTruthLabelID']
        if tf.equal(parton, tf.constant(21)):
            label = 0
        elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
            label = 1
        else:
            label = -1
        return {'jets_pt': data['jets_pt'], 'label': label, 'all_label': parton}

    dataset = dataset.map(get_labeled_pt)
    df = tfds.as_dataframe(dataset)
    df['Truth Label'] = df['label'].replace({1: 'quark', 0: 'gluon'})
    df['All Truth Label'] = df['all_label'].replace({1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 21: 'g'})
    sns.histplot(data=df, x='jets_pt', hue='Truth Label',
                 stat='count', element="step", fill=True,
                 palette='Set1', common_norm=False, hue_order=['quark', 'gluon'])
    plt.savefig(save_path)
    plt.yscale('log')
    plt.savefig(save_path.replace('.png', '_log.png'))
    plt.close()

    sns.histplot(data=df, x='jets_pt', hue='All Truth Label',
                 stat='count', element="step", fill=True, multiple='stack',
                 palette='Set1', common_norm=False)
    plt.savefig(save_path.replace('.png', '_all.png'))
    plt.yscale('log')
    plt.savefig(save_path.replace('.png', '_all_log.png'))
    plt.close()
