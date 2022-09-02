import matplotlib.pyplot as plt
import seaborn as sns



def get_pred_hist(labels, path:str):
    figure = plt.figure(figsize=(8, 8))
    plt.hist(labels, bins=100, label='predictions', alpha=0.5)
    plt.xlabel('Prediction probability')
    plt.ylabel('Count')    
    figure.savefig(path)

def get_pred_violin(labels, path:str):
    figure = plt.figure(figsize=(8, 8))
    sns.violinplot(data=labels)
    plt.xlabel('Prediction probability')
    plt.ylabel('Count')    
    figure.savefig(path)