Module jidenn.evaluation.plotter
================================
Module for plotting the results of the evaluation and plotting the training history.
The ploting is done by subsclassing the `jidenn.evaluation.plotter.ValidationFigure` class,
with the `jidenn.evaluation.plotter.ValidationFigure.get_fig` method being the main method.

Each figure is saved as a png file, as a csv file, and saved to tensorboard.

Functions
---------

    
`plot_metrics_per_cut(df: pandas.core.frame.DataFrame, logdir: str, log: logging.Logger, formats=['png', 'pdf'])`
:   Plots the metrics for different cuts and saves them to disk.
    
    Args:
        df (pd.DataFrame): dataframe containing the metrics for different cuts. 
        logdir (str): The directory where the figures are saved.
        log (Logger): The logger.
        formats (list, optional): The formats in which the figures are saved. Defaults to ['jpg', 'pdf'].

    
`plot_train_history(data: List[float], logdir: str, name: str, epochs: int)`
:   Plots the training history and saves it to disk.
    
    Args:
        data (list): The metric values as a function of the epoch.
        logdir (str): The directory where the figures are saved.
        name (str): The name of the metric.
        epochs (int): The number of epochs.

    
`plot_validation_figs(df: pandas.core.frame.DataFrame, logdir: str, log: logging.Logger, formats: List[str] = ['jpg', 'pdf'])`
:   Plots the validation figures and saves them to disk.
    Args:
        df (pd.DataFrame): The dataframe containing the truth lables, the model output scores and the predictions.
        logdir (str): The directory where the figures are saved.
        log (Logger): The logger.
        formats (list, optional): The formats in which the figures are saved. Defaults to ['jpg', 'pdf'].

Classes
-------

`ValidationCM(df: pandas.core.frame.DataFrame, name: str = 'fig', class_names: Optional[List[str]] = None)`
:   Plots the confusion matrix.

    ### Ancestors (in MRO)

    * jidenn.evaluation.plotter.ValidationFigure

    ### Instance variables

    `figure`
    :   Returns the matplotlib figure.

    `figure_name`
    :   Returns the name of the figure.

    ### Methods

    `get_fig(self, fig: Optional[matplotlib.figure.Figure] = None) ‑> matplotlib.figure.Figure`
    :   Creates matplotlib figure.

    `save_data(self, path: str)`
    :   Saves the data to the specified path as a csv file.

    `save_fig(self, path: str, format: str = 'png')`
    :   Saves the figure to the specified path.

    `to_tensorboard(self, path: str)`
    :   Saves the figure to tensorboard.

`ValidationFigure(df: pandas.core.frame.DataFrame, name: str = 'fig', class_names: Optional[List[str]] = None)`
:   Base class for validation figures.
    
    Only the `get_fig` method needs to be implemented.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        name (str, optional): The name of the figure. Defaults to 'fig'.
        class_names (List[str], optional): The names of the classes. Defaults to None.

    ### Descendants

    * jidenn.evaluation.plotter.ValidationCM
    * jidenn.evaluation.plotter.ValidationLabelHistogram
    * jidenn.evaluation.plotter.ValidationROC
    * jidenn.evaluation.plotter.ValidationScoreHistogram

    ### Instance variables

    `figure`
    :   Returns the matplotlib figure.

    `figure_name`
    :   Returns the name of the figure.

    ### Methods

    `get_fig(self, fig: Optional[matplotlib.figure.Figure] = None) ‑> matplotlib.figure.Figure`
    :   Creates matplotlib figure.

    `save_data(self, path: str)`
    :   Saves the data to the specified path as a csv file.

    `save_fig(self, path: str, format: str = 'png')`
    :   Saves the figure to the specified path.

    `to_tensorboard(self, path: str)`
    :   Saves the figure to tensorboard.

`ValidationLabelHistogram(df: pandas.core.frame.DataFrame, name: str = 'fig', class_names: Optional[List[str]] = None)`
:   Plots the predicted labels of the model, colored by the truth label.
    This is usefull to see if the model is biased towards one class.

    ### Ancestors (in MRO)

    * jidenn.evaluation.plotter.ValidationFigure

    ### Instance variables

    `figure`
    :   Returns the matplotlib figure.

    `figure_name`
    :   Returns the name of the figure.

    ### Methods

    `get_fig(self, fig: Optional[matplotlib.figure.Figure] = None) ‑> matplotlib.figure.Figure`
    :   Creates matplotlib figure.

    `save_data(self, path: str)`
    :   Saves the data to the specified path as a csv file.

    `save_fig(self, path: str, format: str = 'png')`
    :   Saves the figure to the specified path.

    `to_tensorboard(self, path: str)`
    :   Saves the figure to tensorboard.

`ValidationROC(df: pandas.core.frame.DataFrame, name: str = 'fig', class_names: Optional[List[str]] = None)`
:   Class for plotting the ROC curve.
    The ROC curve is calculated using the `sklearn.metrics.roc_curve` function.

    ### Ancestors (in MRO)

    * jidenn.evaluation.plotter.ValidationFigure

    ### Instance variables

    `figure`
    :   Returns the matplotlib figure.

    `figure_name`
    :   Returns the name of the figure.

    ### Methods

    `get_fig(self, fig: Optional[matplotlib.figure.Figure] = None) ‑> matplotlib.figure.Figure`
    :   Creates matplotlib figure.

    `save_data(self, path: str)`
    :   Saves the data to the specified path as a csv file.

    `save_fig(self, path: str, format: str = 'png')`
    :   Saves the figure to the specified path.

    `to_tensorboard(self, path: str)`
    :   Saves the figure to tensorboard.

`ValidationScoreHistogram(df: pandas.core.frame.DataFrame, name: str = 'fig', class_names: Optional[List[str]] = None)`
:   Plots the output scores of the model, colored by the truth label.

    ### Ancestors (in MRO)

    * jidenn.evaluation.plotter.ValidationFigure

    ### Instance variables

    `figure`
    :   Returns the matplotlib figure.

    `figure_name`
    :   Returns the name of the figure.

    ### Methods

    `get_fig(self, fig: Optional[matplotlib.figure.Figure] = None) ‑> matplotlib.figure.Figure`
    :   Creates matplotlib figure.

    `save_data(self, path: str)`
    :   Saves the data to the specified path as a csv file.

    `save_fig(self, path: str, format: str = 'png')`
    :   Saves the figure to the specified path.

    `to_tensorboard(self, path: str)`
    :   Saves the figure to tensorboard.