import os
import shutil
import argparse
import json
import subprocess

from utils.const import MODEL_NAMING_SCHEMA, METRIC_NAMING_SCHEMA

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="Directory.")
parser.add_argument("--title", type=str, help="Subtitle.")


def main(args):
    logdir = args.dir
    save_dir = os.path.join(logdir, 'latex')
    model_names = ['interacting_depart', 'interacting_part', 'highway',
                   'fc', 'transformer', 'part', 'depart', 'pfn', 'efn']

    if not os.path.exists(os.path.join(logdir, 'figs')):
        raise ValueError('No figs folder found')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'figs'), exist_ok=True)

    big_string = ''
    big_string += r'\documentclass[xcolor=table]{beamer}' + '\n'
    big_string += r'\usetheme{Boadilla}' + '\n'
    big_string += r'\usepackage[export]{adjustbox}' + '\n'
    big_string += r'\usepackage[table]{xcolor}' + '\n'
    big_string += r'\usepackage{hyperref}' + '\n'
    big_string += r'\usepackage{subcaption}' + '\n'
    big_string += r'\usepackage{upgreek}' + '\n'
    big_string += r'\usepackage{booktabs}' + '\n'
    big_string += r'\setbeamerfont{title}{size=\Huge}' + '\n'
    big_string += r'\setbeamertemplate{footline}' + '\n'
    big_string += r'{' + '\n'
    big_string += r'  \leavevmode%' + '\n'
    big_string += r'  \hbox{%' + '\n'
    big_string += r'  \begin{beamercolorbox}[wd=.25\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%' + '\n'
    big_string += r'    \usebeamerfont{title in head/foot}\insertshorttitle\hspace*{3em}' + '\n'
    big_string += r'  \end{beamercolorbox}%' + '\n'
    big_string += r'  \begin{beamercolorbox}[wd=.65\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%' + '\n'
    big_string += r'    \usebeamerfont{author in head/foot}\insertinstitute' + '\n'
    big_string += r'  \end{beamercolorbox}%' + '\n'
    big_string += r'  \begin{beamercolorbox}[wd=.10\paperwidth,ht=2.25ex,dp=1ex,center]{page in head/foot}%' + '\n'
    big_string += r'    \usebeamerfont{author in head/foot}\insertframenumber{} / \inserttotalframenumber\hspace*{1ex}' + '\n'
    big_string += r'  \end{beamercolorbox}}%' + '\n'
    big_string += r'  \vskip0pt%' + '\n'
    big_string += r'}' + '\n'
    big_string += r'' + '\n'
    big_string += r'\title{' + args.title + '}\n'
    big_string += r'\subtitle{JIDENN}' + '\n'
    big_string += r'\author{Samuel Jankových, Vojtěch Pleskot}' + '\n'
    big_string += r'\institute{Faculty of Mathematics and Physics, Charles University}' + '\n'
    big_string += r'\date{\today}' + '\n'
    big_string += r'\beamertemplatenavigationsymbolsempty' + '\n'
    big_string += r'' + '\n'
    big_string += r'\begin{document}' + '\n'
    big_string += r'' + '\n'
    big_string += r'\begin{frame}' + '\n'
    big_string += r'  \titlepage' + '\n'
    big_string += r'\end{frame}' + '\n'
    big_string += r'' + '\n'
    big_string += r'' + '\n'
    big_string += r'\begin{frame}' + '\n'
    big_string += r'  METRICS' + '\n'
    big_string += r'\end{frame}' + '\n'
    big_string += r'' + '\n'
    big_string += r'' + '\n'

    for eval_bin in os.listdir(os.path.join(logdir, 'figs')):
        name = eval_bin.split('_')[0]
        for fig_name in os.listdir(os.path.join(logdir, 'figs', eval_bin)):
            if fig_name.endswith('.jpg') or fig_name.endswith('.png') or fig_name.endswith('.pdf'):
                os.makedirs(os.path.join(save_dir, 'figs', eval_bin), exist_ok=True)
                shutil.copy(os.path.join(logdir, 'figs', eval_bin, fig_name),
                            os.path.join(save_dir, 'figs', eval_bin, fig_name))
                fig_name_save = fig_name.replace('_', '_')
                try:
                    fig_name_cap = METRIC_NAMING_SCHEMA[fig_name[:-4]]
                except KeyError:
                    fig_name_cap = fig_name[:-4]
                    print(f'No key {fig_name[:-4]} in {METRIC_NAMING_SCHEMA.keys()}')
                save_eval_bin = eval_bin.replace('_', '_')
                name = name.replace('_', ' ')

                width = '0.6' if fig_name[:-4] == 'heatmap' else '0.8'

                big_string += f'\\begin{{frame}}{{{name} dependency}}\n'
                big_string += f'\\begin{{figure}}[H]\n'
                big_string += f'    \\centering\n'
                big_string += f'    \\includegraphics[width={width}\\textwidth]{{figs/{save_eval_bin}/{fig_name_save}}}\n'
                big_string += f'    \\caption{{{fig_name_cap}}}\n'
                big_string += f'    \\label{{fig:{fig_name_save[:-4]}_{eval_bin}}}\n'
                big_string += f'\\end{{figure}}\n'
                big_string += f'\\end{{frame}}\n\n'

    fig_names = os.listdir(os.path.join(logdir, model_names[0], 'eval_pT', 'base', 'figs', 'pdf'))
    fig_names = [fig_name for fig_name in fig_names if fig_name.endswith('.pdf')]
    for fig_name in fig_names:
        i = 0
        for model in model_names:
            os.makedirs(os.path.join(save_dir, 'figs', model), exist_ok=True)
            shutil.copy(os.path.join(logdir, model, 'eval_pT', 'base', 'figs', 'pdf', fig_name),
                        os.path.join(save_dir, 'figs', model, fig_name))

            save_model = model.replace('_', '_')
            save_fig_name = fig_name.replace('_', '_')
            model = model.replace('_', ' ')
            cap_fig_name = fig_name.replace('_', ' ')[:-4]

            if i % 2 == 0:
                big_string += f'\\begin{{frame}}{{{cap_fig_name}}}\n'
                big_string += f'\\begin{{figure}}[htb]\n'
                big_string += f'    \\centering\n'

            big_string += f'\\begin{{subfigure}}[t]{{0.49\\textwidth}}\n'
            big_string += f'    \\centering\n'
            big_string += f'    \\includegraphics[width=1.\\textwidth]{{figs/{save_model}/{save_fig_name}}}\n'
            big_string += f'    \\caption{{{model}}}\n'
            big_string += f'    \\label{{fig:{save_fig_name[:-4]}_{save_model}}}\n'
            big_string += f'\\end{{subfigure}}\n'

            if i % 2 == 1:
                big_string += f'  \\caption{{{cap_fig_name}}}\n'
                big_string += f'  \\label{{fig:{save_fig_name[:-4]}_{i}}}\n'
                big_string += f'\\end{{figure}}\n'
                big_string += f'\\end{{frame}}\n\n'

            i += 1

        if i % 2 == 1:
            big_string += f'  \\caption{{{cap_fig_name}}}\n'
            big_string += f'  \\label{{fig:{save_fig_name[:-4]}_{i}}}\n'
            big_string += f'\\end{{figure}}\n'
            big_string += f'\\end{{frame}}\n\n'

    big_string += r'\end{document}'
    try:
        os.remove(os.path.join(logdir, 'metrics.txt'))
    except FileNotFoundError:
        pass

    subprocess.check_call(f"bash_scripts/get_metrics.sh {logdir}", shell=True)
    with open(os.path.join(logdir, 'metrics.txt'), 'r') as f:
        metrics = f.readlines()
    table = ''
    table += r'\begin{table}[htb]' + '\n'
    table += r'  \centering' + '\n'
    rows = ''
    for line in metrics:
        mod, metr = line.split('-')
        metr = json.loads(metr.replace("'", '"').replace('inf', '-1'))
        rows += f'{MODEL_NAMING_SCHEMA[mod]} & ' + ' & '.join([f'{metr[met] :.4f}' for met in metr]) + r'\\' + '\n'
        metric_names = list(metr.keys())

    table += r'  \scalebox{0.6}{'
    table += r'  \begin{tabular}{l' + 'c' * len(metric_names) + '}' + '\n'
    table += r'    \toprule' + '\n'
    table += r'    Model & ' + ' & '.join([f'{METRIC_NAMING_SCHEMA[met]}' for met in metric_names]) + r'\\' + '\n'
    table += r'    \midrule' + '\n'
    table += rows
    table += r'    \bottomrule' + '\n'
    table += r'  \end{tabular}}' + '\n'
    table += r'  \caption{Metrics}' + '\n'
    table += r'  \label{tab:metrics}' + '\n'
    table += r'\end{table}' + '\n\n'

    big_string = big_string.replace('METRICS', table)

    with open(os.path.join(save_dir, 'main.tex'), 'w') as f:
        f.write(big_string)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
