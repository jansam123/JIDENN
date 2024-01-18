import os
import sys
sys.path.append(os.getcwd())
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import pandas as pd
import argparse
# from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA, MC_NAMING_SCHEMA, METRIC_NAMING_SCHEMA_NAMED
MC_NAMING_SCHEMA = {}
METRIC_NAMING_SCHEMA = {}
METRIC_NAMING_SCHEMA_NAMED = {}

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", type=str, nargs='*', help="Paths to the saved evaluation metrics.")
args = parser.parse_args()

args.load_dir = [args.load_dir] if isinstance(args.load_dir, str) else args.load_dir
XLABEL = r'pT [TeV]'
# COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
COLORS = ["#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C", "#98DF8A", "#D62728", "#FF9896", "#9467BD", "#C5B0D5", "#8C564B", "#C49C94", "#E377C2", "#F7B6D2", "#7F7F7F", "#C7C7C7", "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5"]
# random.shuffle(COLORS)


named_dfs = {}
for idx, path in enumerate(args.load_dir):
    model_names = os.listdir(path)
    dfs = []
    for name in model_names:
        model_df = pd.read_csv(os.path.join(path, name, 'binned_metrics.csv'))
        model_df['Model'] = name
        dfs.append(model_df)
        
    df = pd.concat(dfs)
    df['x_err'] = 1e-6*df['bin_width']/2
    df[XLABEL] = df['bin_mid']*1e-6
    df = df.rename(columns=METRIC_NAMING_SCHEMA_NAMED)
    columns = df.columns[1:]
    named_dfs[path] = df


app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='pT dependence of metrics', style={'textAlign':'center'}),
    html.H3(children='Select the models to plot', style={'textAlign':'left'}),
    html.Div([
        dcc.Checklist(list(args.load_dir), value=list(args.load_dir), id='path-list'),
    ]),
    html.Br(),
    html.H3(children='Select the metric to plot', style={'textAlign':'left'}),
    dcc.Dropdown(columns, columns[0], id=f'dropdown-selection'),
    html.Br(),
    html.H3(children='Select the model to use as reference', style={'textAlign':'left'}),
    dcc.RadioItems(
        id=f'radio-items',
        options=list(args.load_dir),
        value=list(args.load_dir)[0],
    ),
    html.Br(),
    dcc.Graph(id=f'graph-content'),
    # html.Br(),
    # html.H2(children='Envelope plot', style={'textAlign':'center'}),
    # html.Br(),
    # dcc.Graph(id=f'envelope-plot'),
])

# @callback(
#     Output('radio-items', 'options'),
#     Input('editing-columns-button', 'n_clicks'),
#     State('editing-columns-name', 'value'),
#     State('radio-items', 'options')
# )
# def add_radio_items(n_clicks, value, options):
#     if 'Please add a path...' in options and value == '': 
#         return ['Please add a path...']
#     elif 'Please add a path...' in options and value != '':
#         options.remove('Please add a path...')
#     return options + [value] if value not in options else options


@callback(
    Output(f'graph-content', 'figure'),
    Input('dropdown-selection', 'value'),
    Input('path-list', 'value'),
    Input('radio-items', 'value')
)
def update_graph(value, paths, reference):
    if paths is not None and len(paths) == 0: 
        return go.Figure()
    rows = len(paths) // 2 + len(paths) % 2
    specs = [[{}, {}]]*rows
    rows += 2
    specs += [[{'colspan': 2, 'rowspan': 2}, None]]
    specs += [[None, None]]
    subtitles = []
    for path in paths:
        subtitles.append(MC_NAMING_SCHEMA[path.split('/')[-2]]) if path.split('/')[-2] in MC_NAMING_SCHEMA.keys() else subtitles.append(path.split('/')[-2])
    
    fig = make_subplots(
    rows=rows, cols=2, subplot_titles=[path.split('/')[-2] for path in paths] + ['Envelope plot'], shared_xaxes=True, shared_yaxes=True, specs=specs,
    vertical_spacing=0.2/rows, horizontal_spacing=0.01, 
    )
    font_dict=dict(family='Arial',
            size=22,
            color='black'
            )
    fig.update_layout(
        height=450*rows,
        font=font_dict,  # font formatting
        plot_bgcolor='white',  # background color
        margin=dict(r=20,t=20,b=10)
    )   
    
    evelope_dfs = []
    for idx, path in enumerate(paths):
        
        if paths is not None and path not in paths: 
            continue
        # fig = px.scatter(named_dfs[idx], x=XLABEL, y=value, color='Model', log_x=False, log_y=False, error_x='x_err')
        # fig.layout.annotations[idx].update(x=0.025) if idx % 2 == 0 else fig.layout.annotations[idx].update(x=0.525)
        
        for model_idx, model in enumerate(named_dfs[path]['Model'].unique()):
            model_df = named_dfs[path][named_dfs[path]['Model'] == model]
            evelope_metric = abs(1 - model_df[value].values/named_dfs[reference][named_dfs[reference]['Model'] == model][value].values)
            envelope_df = pd.DataFrame({'metric': evelope_metric, XLABEL: model_df[XLABEL].values, 'x_err': model_df['x_err'].values})
            envelope_df['model'] = model
            evelope_dfs.append(envelope_df)
            fig.add_trace(
                go.Scatter(
                    x=model_df[XLABEL],
                    y=model_df[value],
                    error_x=dict(
                        type='data',
                        array=model_df['x_err'],
                        visible=True
                    ),
                    mode='markers+lines',
                    marker=dict(
                        size=15,
                        color=COLORS[model_idx%len(COLORS)]
                    ),
                    name=model[:30],
                    showlegend=True if idx == 0 else False,
                    legendgroup=model,
                    
                ),
                row=(idx//2)+1, col=(idx%2)+1
        )
            fig.update_yaxes(
                row=(idx//2)+1, col=(idx%2)+1,
                matches='y'
                )

    fig.update_annotations(font_size=16)
    fig.update_yaxes(
                title_text=value,
                showline=True,  # add line at x=0
                linecolor='black',  # line color
                linewidth=1.8, # line size
                ticks='inside',  # ticks outside axis
                minor_ticks='inside',
                tickfont=font_dict, # tick label font
                mirror='allticks',  # add ticks to top/right axes
                tickwidth=1.8,  # tick width
                ticklen=10,
                minor_ticklen=5,
                tickcolor='black',  # tick color
                # matches='x'
                # matches='y'
                )
    fig.update_xaxes(
                title_text=XLABEL,
                showline=True,
                showticklabels=True,
                linecolor='black',
                linewidth=1.8,
                ticks='inside',
                minor_ticks='inside',
                tickfont=font_dict,
                mirror='allticks',
                tickwidth=1.8,
                ticklen=10,
                minor_ticklen=5,
                tickcolor='black',
                # matches='x'
                title_standoff = 1
                )
    
    evelope_df = pd.concat(evelope_dfs)
    evelope_df = evelope_df.groupby(['model', XLABEL]).max().reset_index()
    for model_idx, model in enumerate(named_dfs[reference]['Model'].unique()):
        local_df = evelope_df[evelope_df['model'] == model]
        fig.add_trace(
            go.Scatter(
                x=local_df[XLABEL],
                y=local_df['metric'],
                mode='markers+lines',
                marker=dict(
                        size=15,
                        color=COLORS[model_idx%len(COLORS)]
                ),
                error_x=dict(
                        type='data',
                        array=local_df['x_err'],
                        visible=True
                ),
                name=model,
                showlegend=False,
                legendgroup=model,
            ),
            row=rows-1, col=1
        )
        
    fig.update_yaxes(
        col=1, row=rows-1,
        title_text='max over models of |1 - metric of model/metric of ref model|',
    )

    return fig




if __name__ == '__main__':
    app.run(debug=True)