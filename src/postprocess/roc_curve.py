from  sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def get_roc(labels, predictions, path, plot_label=None):
    fp, tp, _ = roc_curve(labels, predictions)
    
    figure = plt.figure(figsize=(8, 8))
    plt.plot(100*fp, 100*tp, label=plot_label, linewidth=2)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    # plt.xlim([-0.5,20])
    # plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    figure.savefig(path)