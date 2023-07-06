"""
Module for plotting the results of the evaluation and plotting the training history.
The ploting is done by subsclassing the `jidenn.evaluation.plotter.ValidationFigure` class,
with the `jidenn.evaluation.plotter.ValidationFigure.get_fig` method being the main method.

Each figure is saved as a png file, as a csv file, and saved to tensorboard.
"""
from logging import Logger
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
from io import BytesIO
from typing import List, Union, Dict, Optional, Tuple
import atlasify
import puma


sns.set_theme(style="ticks")


class ValidationFigure:
    """Base class for validation figures.

    Only the `get_fig` method needs to be implemented.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        name (str, optional): The name of the figure. Defaults to 'fig'.
        class_names (List[str], optional): The names of the classes. Defaults to None.

    """

    def __init__(self, df: pd.DataFrame, name: str = 'fig', class_names: Union[List[str], None] = None):
        self._df = df
        self._name = name
        self._class_names = class_names
        self._data = None
        self._fig = self.get_fig()

    @property
    def figure(self):
        """Returns the matplotlib figure."""
        return self._fig

    @property
    def figure_name(self):
        """Returns the name of the figure."""
        return self._name

    @abstractmethod
    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        "Creates matplotlib figure."

    def save_fig(self, path: str, format: str = 'png'):
        """Saves the figure to the specified path."""
        self._fig.savefig(os.path.join(path, self._name +
                          f".{format}"), dpi=300, bbox_inches='tight')

    def save_data(self, path: str):
        """Saves the data to the specified path as a csv file."""
        if self._data is None:
            return
        self._data.to_csv(os.path.join(path, self._name + ".csv"))

    def to_tensorboard(self, path: str):
        """Saves the figure to tensorboard."""
        with tf.summary.create_file_writer(path).as_default():
            tf.summary.image(self._name, self._fig_to_image(self._fig), step=0)

    def _fig_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        buf = BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        figure.savefig(buf, format='png')

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        # plt.close(figure)
        buf.seek(0)
        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


class ValidationROC(ValidationFigure):
    """Class for plotting the ROC curve.
    The ROC curve is calculated using the `sklearn.metrics.roc_curve` function.
    """

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        fp, tp, th = roc_curve(
            self._df['label'].values, self._df['score'].values)
        self._data = pd.DataFrame({'FPR': fp, 'TPR': tp, 'threshold': th})
        auc_score = auc(fp, tp)

        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        sns.lineplot(x=100 * fp, y=100 * tp,
                     label=f'AUC = {auc_score:.3f}', linewidth=2)
        sns.lineplot(x=[0, 50, 100], y=[0, 50, 100], label=f'Random',
                     linewidth=1, linestyle='--', color='darkred', alpha=0.5)
        plt.plot([0, 0, 100], [0, 100, 100], color='darkgreen',
                 linestyle='-.', label='Ideal', alpha=0.5)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal')
        return fig


class ValidationCM(ValidationFigure):
    """Plots the confusion matrix."""

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        cm = confusion_matrix(
            self._df['label'].values, self._df['prediction'].values)
        if fig is None:
            fig = plt.figure(figsize=(6, 6))

        cm = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis] * 1000, decimals=0).astype(int)
        df_cm = pd.DataFrame(cm, index=self._class_names,
                             columns=self._class_names)
        self._data = df_cm
        sns.heatmap(df_cm, annot=True, fmt='4d', cmap=plt.cm.Blues, cbar=False)
        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig


class ValidationScoreHistogram(ValidationFigure):
    """Plots the output scores of the model, colored by the truth label."""

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        self._data = self._df[['score', 'Truth Label']]
        ax = sns.histplot(data=self._df, x='score', hue='Truth Label',
                          palette='Set1', stat='count', element="step", fill=True,
                          hue_order=self._class_names)
        sns.move_legend(ax, 'upper center')
        plt.xlabel('Score')
        return fig


class ValidationLabelHistogram(ValidationFigure):
    """Plots the predicted labels of the model, colored by the truth label.
    This is usefull to see if the model is biased towards one class.
    """

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        self._data = self._df[['Truth Label', 'named_prediction']]
        sns.histplot(self._df, x='named_prediction', hue='Truth Label',
                     stat='count', multiple='stack', hue_order=self._class_names,
                     palette='Set1')
        plt.xlabel('Predicted Tag')
        return fig


def plot_validation_figs(df: pd.DataFrame,
                         logdir: str,
                         formats: List[str] = ['jpg', 'pdf'],
                         class_names: Optional[List[str]] = None,
                         log: Optional[Logger] = None,
                         score_name: str = 'score'):
    """Plots the validation figures and saves them to disk.
    Args:
        df (pd.DataFrame): The dataframe containing the truth lables, the model output scores.
        logdir (str): The directory where the figures are saved.
        log (Logger): The logger.
        formats (list, optional): The formats in which the figures are saved. Defaults to ['jpg', 'pdf'].
        score_name (str, optional): The name of the model output score. Defaults to 'score'.

    """
    if score_name != 'score':
        df = df.rename(columns={score_name: 'score'})

    df['prediction'] = df['score'].apply(lambda x: 1 if x > 0.5 else 0)
    df['Truth Label'] = df['label'].apply(
        lambda x: class_names[x]) if class_names is not None else df['label'].apply(str)
    df['named_prediction'] = df['prediction'].apply(
        lambda x: class_names[x]) if class_names is not None else df['prediction'].apply(str)

    base_path = os.path.join(logdir, "figs")
    tb_base_path = os.path.join(logdir, "plots")
    csv_path = os.path.join(base_path, 'csv')
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(tb_base_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)
    format_path = []
    for format in formats:
        format_path.append(os.path.join(base_path, format))
        os.makedirs(format_path[-1], exist_ok=True)

    figure_classes = [ValidationROC, ValidationCM,
                      ValidationScoreHistogram, ValidationLabelHistogram]
    figure_names = ['roc', 'confusion_matrix', 'score_hist', 'prediction_hist']

    for validation_fig, name in zip(figure_classes, figure_names):
        log.info(f"Generating figure {name}") if log else None
        val_fig = validation_fig(df, name, class_names=class_names)
        for fmt, path in zip(formats, format_path):
            val_fig.save_fig(path, fmt)
        val_fig.save_data(csv_path)
        val_fig.to_tensorboard(tb_base_path)
    plt.close('all')


def plot_metrics_per_cut(df: pd.DataFrame, logdir: str, log: Logger, formats=['png', 'pdf']):
    """Plots the metrics for different cuts and saves them to disk.

    Args:
        df (pd.DataFrame): dataframe containing the metrics for different cuts. 
        logdir (str): The directory where the figures are saved.
        log (Logger): The logger.
        formats (list, optional): The formats in which the figures are saved. Defaults to ['jpg', 'pdf'].
    """
    base_path = os.path.join(logdir, "metrics")
    os.makedirs(base_path, exist_ok=True)
    format_path = []
    for format in formats:
        format_path.append(os.path.join(base_path, format))
        os.makedirs(format_path[-1], exist_ok=True)

    for metric in df.columns:
        if metric == 'cut':
            continue
        log.info(f"Plotting {metric} for cuts")
        sns.pointplot(x='cut', y=metric, data=df, join=False)
        plt.xlabel('Cut')
        plt.ylabel(str(metric))
        for fmt, path in zip(formats, format_path):
            plt.savefig(os.path.join(path, f'{metric}.{fmt}'))
        plt.close()


def plot_train_history(data: List[float], logdir: str, name: str, epochs: int):
    """Plots the training history and saves it to disk.

    Args:
        data (list): The metric values as a function of the epoch.
        logdir (str): The directory where the figures are saved.
        name (str): The name of the metric.
        epochs (int): The number of epochs.
    """
    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=data, linewidth=2.5, palette='husl')
    g.set(xlabel='Epoch', ylabel=name)
    g.set_xticks(range(epochs))
    g.set_xticklabels(range(1, epochs + 1))
    plt.grid(True)
    fig.savefig(f'{logdir}/{name}.png')
    plt.close()


def explode_nested_variables(df: pd.DataFrame, exploding_column: str, max_iterations: int = 5) -> pd.DataFrame:
    """Explode a DataFrame by a column containing nested lists or arrays. 
    This allows to plot the distributions of the variables in the nested lists or arrays, such
    as the jet constituents.

    Args:
        df (pd.DataFrame): DataFrame to explode.
        exploding_column (str): Name of the column containing the nested lists or arrays.
        max_iterations (int, optional): Maximum number of iterations to perform. If the column still contains
            non-numeric values after this many iterations, the function will raise a ValueError. Default is 5.
            Set to higher values if the nesting is very deep.

    Returns:
        pd.DataFrame: Exploded DataFrame.

    Raises:
        ValueError: If the column still contains non-numeric values after the maximum number of iterations.

    """
    for _ in range(max_iterations):
        try:
            df[exploding_column] = pd.to_numeric(df[exploding_column])
            break
        except (ValueError, TypeError):
            df = df.explode(exploding_column, ignore_index=True)
            df = df.sample(n=len(df.index)).reset_index(drop=True)
            continue
    return df


def plot_data_distributions(df: pd.DataFrame,
                            folder: str,
                            hue_variable: str = 'label',
                            variables: Optional[List[str]] = None,
                            named_labels: Optional[Dict[int, str]] = None,
                            weight_variable: Optional[str] = None,
                            n_bins: Optional[int] = 100,
                            xlabel_mapper: Optional[Dict[str, str]] = None) -> None:
    r"""Plot the data distributions of the variables in a DataFrame for different truth values.

    Args:
        df (pd.DataFrame): DataFrame containing the input variables to plot and a column named 'label'
            containing the truth values.
        folder (str): Path to the directory where the plots will be saved.
        hue_variable (str, optional): Name of the column containing a categorical or discrete variable
            to use for the hue. Default is 'label'.
        named_labels (Dict[int, str], optional): Dictionary mapping truth values to custom labels.
            If not provided, the truth values will be used as labels. 
        xlabel_mapper (Dict[str, str], optional): Dictionary mapping variable names to custom x-axis labels.
            If not provided, the variable names will be used as labels. This is useful to convert stadarized
            variable names such as 'jets_pt' to a latex formatted label such as '$p_{\mathrm{T}}^\mathrm{jet}$'.

    """
    # hue_order = named_labels.values() if named_labels is not None else None
    df['Truth Label'] = df[hue_variable].apply(
        lambda x: named_labels[x]) if named_labels is not None else df[hue_variable].apply(str)
    hue_order = set(named_labels.values()) if named_labels is not None else None
    color_column = 'Truth Label'
    var_names = list(df.columns) if variables is None else variables
    var_names.remove(color_column) if color_column in var_names else None
    os.makedirs(os.path.join(folder, 'jpg'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'pdf'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'jpg_log'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'pdf_log'), exist_ok=True)
    iter_var_names = var_names + [hue_variable, 'weight'] if weight_variable is not None else var_names + [hue_variable]
    for var_name in iter_var_names:
        small_df = df[[var_name, color_column]].copy()
        dtype = small_df[var_name].dtype

        if dtype == 'object':
            small_df = explode_nested_variables(small_df, var_name)
            small_df = small_df.loc[small_df[var_name] != 0]
        try:
            ax = sns.histplot(data=small_df, x=var_name, hue=color_column,
                              stat='density', element="step", fill=True,
                              palette='Set1', common_norm=False, hue_order=hue_order)
        except:
            ax = sns.histplot(data=small_df, x=var_name, hue=color_column,
                              stat='density', element="step", fill=True,
                              palette='Set1', common_norm=False, hue_order=hue_order, bins=n_bins if n_bins is not None else 'auto')

        plt.xlabel(xlabel_mapper[var_name] if xlabel_mapper is not None and var_name in xlabel_mapper else var_name)

        atlasify.atlasify(subtext="Simulation Internal")
        plt.savefig(os.path.join(folder, 'jpg', f'{var_name}.jpg'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(folder, 'pdf', f'{var_name}.pdf'), bbox_inches='tight')
        plt.yscale('log')
        atlasify.atlasify(subtext="Simulation Internal")
        plt.savefig(os.path.join(folder, 'jpg_log', f'{var_name}.jpg'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(folder, 'pdf_log', f'{var_name}.pdf'), bbox_inches='tight')

        plt.close()


def plot_single_dist(df: pd.DataFrame,
                     variable: str,
                     hue_var: str = 'label',
                     ylog: bool = False,
                     hue_order: Optional[List[str]] = None,
                     save_path: str = 'figs.png') -> None:

    sns.histplot(data=df, x=variable, hue=hue_var,
                 stat='count', element="step", fill=True,
                 palette='Set1', common_norm=False, hue_order=hue_order)
    plt.savefig(save_path)
    plt.yscale('log') if ylog else None
    plt.savefig(save_path)
    plt.close()

    # sns.histplot(data=df, x=variable, hue='All Truth Label',
    #              stat='count', element="step", fill=True, multiple='stack',
    #              palette='Set1', common_norm=False)
    # plt.savefig(save_path.replace('.png', '_all.png'))
    # plt.yscale('log')
    # plt.savefig(save_path.replace('.png', '_all_log.png'))
    # plt.close()


def plot_var_dependence(dfs: List[pd.DataFrame],
                        labels: List[str],
                        bin_midpoint_name: str,
                        bin_width_name: str,
                        metric_names: List[str],
                        save_path: str,
                        ratio_reference_label: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel_mapper: Optional[Dict[str, str]] = None,
                        ylims: Optional[List[Tuple[float, float]]] = None,
                        colours: Optional[List[str]] = None,
                        ):
    """Plot the dependence of multiple metrics on a variable in a DataFrame for multiple models.
    The Dataframe must contain columns corresponding to individual metrics, and columns containing
    the information about the binning of the variable. The binning information must be in the form
    of the bin midpoints and bin widths.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrames containing the data to plot. Each DataFrame is plotted
            as a separate line (e.g. ML model comparison or MC simluations comparison).
        labels (List[str]): Names of the labels for each DataFrame displayed in the legend.
        bin_midpoint_name (str): Name of the column containing the bin midpoints.
        bin_width_name (str): Name of the column containing the bin widths.
        metric_names (List[str]): List of names of the metrics to plot inside **each** DataFrame.
        save_path (str): Path to the directory where the plots will be saved.
        ratio_reference_label (str, optional): Label of the DataFrame to use as reference for the ratio plot.
            If not provided, no ratio plot will be drawn. Default is None.
        xlabel (str, optional): Label for the x-axis. 
        ylabel_mapper (Dict[str, str], optional): Dictionary mapping metric names to custom y-axis labels.
            If not provided, the metric names will be used as labels.
        ylims (List[Tuple[float, float]], optional): List of y-axis limits for each metric plot.
            If not provided, the limits will be automatically determined. Default is None.
        colours (List[str], optional): List of colours to use for each label. If not provided, the default
            colour cycle will be used.

    """

    for i, metric_name in enumerate(metric_names):

        plot = puma.VarVsVarPlot(
            ylabel=ylabel_mapper[metric_name] if ylabel_mapper is not None and metric_name in ylabel_mapper else metric_name,
            xlabel=xlabel,
            logy=False,
            ymin=ylims[i][0] if ylims is not None else None,
            ymax=ylims[i][1] if ylims is not None else None,
            n_ratio_panels=1 if ratio_reference_label is not None else 0,
            figsize=(9, 7),
            atlas_second_tag='13 TeV'
        )

        for df, label in zip(dfs, labels):
            x_var = df[bin_midpoint_name].to_numpy()
            x_width = df[bin_width_name].to_numpy()
            y_var_mean = df[metric_name].to_numpy()
            y_var_std = np.zeros_like(y_var_mean)
            # np.sqrt(y_var_mean * (1 - y_var_mean) / df['num_events'].to_numpy())
            plot.add(
                puma.VarVsVar(
                    x_var=x_var,
                    x_var_widths=x_width,
                    y_var_mean=y_var_mean,
                    y_var_std=y_var_std,
                    plot_y_std=False,
                    marker='o',
                    markersize=20,
                    markeredgewidth=20,
                    is_marker=True,
                    label=label,
                    colour=colours[labels.index(label)] if colours is not None else None,
                    # linestyle='-',
                ),
                reference=True if ratio_reference_label is not None and label == ratio_reference_label else False,
            )
        plot.draw()
        plot.savefig(os.path.join(save_path, f'{metric_name}.png'), dpi=300)
