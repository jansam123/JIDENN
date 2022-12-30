import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List


sns.set_theme(style="dark")


def tb_postprocess(data: Dict[str, List[float]], logdir: str, name: str, epochs: int):
    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=data, linewidth=2.5, palette='husl')
    g.set(xlabel='Epoch', ylabel=name)
    g.set_xticks(range(epochs))
    g.set_xticklabels(range(1, epochs+1))
    plt.grid(True)
    fig.savefig(f'{logdir}/{name}.png')
    plt.close()