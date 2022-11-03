import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="dark")
plt.style.use('seaborn-darkgrid')

# loss = [0.551, 0.5477, 0.5464, 0.5459, 0.5461, 0.5463, 0.5457, 0.548, 0.5523, 0.5476,
#         0.5449, 0.5437, 0.5432, 0.5428, 0.5428, 0.5422, 0.542, 0.5419, 0.5418, 0.5417, ]
# accuracy = [0.7142, 0.7161, 0.7172, 0.7178, 0.7171, 0.7172, 0.7179, 0.7167, 0.7129,
#             0.7171, 0.7186, 0.7192, 0.7196, 0.7198, 0.72, 0.7202, 0.7203, 0.7204, 0.7205, 0.7205, ]
# auc = [0.7905, 0.7931, 0.7944, 0.7949, 0.7945, 0.7944, 0.795, 0.7931, 0.789, 0.7935,
#        0.7958, 0.7967, 0.7972, 0.7975, 0.7978, 0.798, 0.7982, 0.7983, 0.7984, 0.7984]

# val_loss = [0.549, 0.5473, 0.5456, 0.5451, 0.5473, 0.545, 0.5449, 0.5518, 0.5458, 0.5472,
#             0.5452, 0.543, 0.5429, 0.5427, 0.5424, 0.5422, 0.5421, 0.5419, 0.5418, 0.5418, ]
# val_accuracy = [0.7158, 0.7169, 0.7182, 0.7186, 0.716, 0.7185, 0.7184, 0.716, 0.7178,
#                 0.7165, 0.7184, 0.7196, 0.7197, 0.7199, 0.72, 0.7201, 0.7203, 0.7204, 0.7204, 0.7205]
# val_auc = [0.7929, 0.7943, 0.7956, 0.7959, 0.7936, 0.7961, 0.796, 0.7906, 0.7961, 0.7958,
#            0.7959, 0.7974, 0.7976, 0.7978, 0.798, 0.7981, 0.7982, 0.7983, 0.7984, 0.7984]

# accuracy = [0.7154, 0.7174, 0.718, 0.7183, 0.7186, 0.7188, 0.719, 0.7192, 0.7193, 0.7195,
#             0.7197, 0.7198, 0.72, 0.7201, 0.7202, 0.7203, 0.7204, 0.7205, 0.7206, 0.7206]

# loss = [0.549, 0.5462, 0.5456, 0.545, 0.5446, 0.5457, 0.5445, 0.5439, 0.5437, 0.5434,
#         0.5431, 0.5429, 0.5428, 0.5426, 0.5437, 0.5431, 0.542, 0.5419, 0.5418, 0.5417]
# auc = [0.792, 0.7945, 0.7952, 0.7957, 0.796, 0.7962, 0.7964, 0.7967, 0.7969, 0.7971,
#        0.7973, 0.7975, 0.7976, 0.7978, 0.798, 0.7981, 0.7982, 0.7984, 0.7984, 0.7985]

# val_loss = [0.5464, 0.5456, 0.5448, 0.5445, 0.5442, 0.5439, 0.5437, 0.5435, 0.5433,
#             0.5431, 0.543, 0.5428, 0.5426, 0.5425, 0.5424, 0.5423, 0.5421, 0.5421, 0.542, 0.542]
# val_accuracy = [0.7173, 0.7178, 0.7185, 0.7187, 0.7188, 0.7189, 0.7191, 0.7193, 0.7195,
#                 0.7196, 0.7197, 0.7198, 0.72, 0.72, 0.7201, 0.7202, 0.7203, 0.7204, 0.7205, 0.7205]
# val_auc = [0.7945, 0.7952, 0.796, 0.7962, 0.7965, 0.7967, 0.7968, 0.797, 0.7972, 0.7974,
#            0.7975, 0.7976, 0.7978, 0.7979, 0.798, 0.7981, 0.7982, 0.7983, 0.7983, 0.7984]


def epochs_plot():
    accuracy = [0.716, 0.7181, 0.7187, 0.7191, 0.7195, 0.7196, 0.7194, 0.7202, 0.7207, 0.7209]

    loss = [0.5481, 0.5452, 0.5443, 0.5438, 0.5435, 0.5444, 0.5446, 0.5423, 0.5415, 0.5412]

    auc = [0.7928, 0.7954, 0.7962, 0.7967, 0.7972, 0.7972, 0.7969, 0.798, 0.7986, 0.7989]

    val_loss = [0.5457, 0.5448, 0.5439, 0.5434, 0.5432, 0.5431, 0.5431, 0.5421, 0.5418, 0.5417]
    val_accuracy = [0.7178, 0.7186, 0.7191, 0.7195, 0.7197, 0.7196, 0.7197, 0.7202, 0.7205, 0.7205]
    val_auc = [0.7953, 0.796, 0.7967, 0.7971, 0.7973, 0.7974, 0.7974, 0.7982, 0.7984, 0.7985]

    df = pd.DataFrame({'Train Loss': loss, 'Train Accuracy': accuracy, 'Validation Loss': val_loss,
                       'Validation Accuracy': val_accuracy, 'Train AUC': auc, 'Validation AUC': val_auc})

    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=df[['Train Loss', 'Validation Loss']], linewidth=2.5, palette='husl')
    g.set(xlabel='Epoch', ylabel='Loss')
    g.set_xticks(range(len(df)))
    g.set_xticklabels(range(1, len(df)+1))
    plt.grid(True)
    fig.savefig('loss.png')
    plt.close()

    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=df[['Train Accuracy', 'Validation Accuracy']], linewidth=2.5, palette='husl')
    g.set(xlabel='Epoch', ylabel='Accuracy')
    g.set_xticks(range(len(df)))
    g.set_xticklabels(range(1, len(df)+1))
    plt.grid(True)
    fig.savefig('accuracy.png')
    plt.close()

    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=df[['Train AUC', 'Validation AUC']], linewidth=2.5, palette='husl')
    g.set(xlabel='Epoch', ylabel='AUC')
    g.set_xticks(range(len(df)))
    g.set_xticklabels(range(1, len(df)+1))
    plt.grid(True)
    fig.savefig('auc.png')
    plt.close()


def pt_spec_plot():
    accuracy = [0.6362, 0.7340, 0.7639, 0.7766, 0.7802]
    auc = [0.6841, 0.8058, 0.8409, 0.8563, 0.8614]
    p_t = [40, 110, 280, 600, 1100]
    p_t_err = [20, 50, 120, 200, 300]
    df = pd.DataFrame({'Accuracy': accuracy, 'AUC': auc, 'p_t': p_t, 'p_t_err': p_t_err})
    sns.color_palette("husl", 8)
    df['pt_max'] = df['p_t'] + df['p_t_err']
    plt.errorbar(x=df['p_t'], y=df['Accuracy'], xerr=df['p_t_err'],
                 fmt='o',  label='Accuracy', capsize=5, markersize=8, color='darkviolet')
    plt.xlabel('$p_T$ [GeV]')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs $p_T$')
    plt.savefig('accuracy_pT.png')
    plt.close()
    plt.errorbar(x=df['p_t'], y=df['AUC'], xerr=df['p_t_err'], fmt='o',
                 label='AUC', capsize=5, markersize=8, color='darkviolet')
    plt.xlabel('$p_T$ [GeV]')
    plt.ylabel('AUC')
    plt.title('AUC vs $p_T$')
    plt.savefig('auc_pT.png')
    plt.close()


if __name__ == '__main__':
    pt_spec_plot()
