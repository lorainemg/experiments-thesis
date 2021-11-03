import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import get_plot_folder
from functools import reduce
from pathlib import Path
import json
import re


def build_info(file_name, info, i):
    data = {key: value for key, value in info.items() if not isinstance(value, list)}
    # data['failed_pipelines'] = data['failed_pipelines'] / len(info['scores'])
    data['dataset'] = file_name
    data['i'] = i
    if 'max_idx' not in data:
        try:
            max_idx = info['scores'].index(data['best_fn'])
        except:
            max_idx = None
        data['max_idx'] = max_idx
    return data

def extract_scores(metalearner_path: Path):
    "Extracts a feature from an especific task info"
    data_dict = {}
    file_re = re.compile('\w+_(\d+)_(\d+)\.json')
    for fn in metalearner_path.glob('*.json'):
        info = json.load(open(fn, 'r+'))
        m = file_re.match(fn.name)
        dataset_name = m.group(1)
        iteration = int(m.group(2))

        try:
            data_dict[iteration].append(build_info(dataset_name, info, iteration))
        except KeyError:
            data_dict[iteration] = [build_info(dataset_name, info, iteration)]

    return [pd.DataFrame(data) for data in data_dict.values()]


def build_average_dataframe(dfs, metalearner):
    datasets = {}
    for df in dfs:
        for _, row in df.iterrows():
            dataset = row['dataset']
            for column in df.columns:
                if column == 'dataset' or column == 'i':
                    continue
                try:
                    datasets[dataset][column].append(row[column])
                except KeyError:
                    try:
                        datasets[dataset][column] = [row[column]]
                    except KeyError:
                        datasets[dataset] = {column: [row[column]]}
    aggregate_datasets = []
    for ds, values in datasets.items():
        ds_dict = {prop: np.mean(list_v) for prop, list_v in values.items()}
        aggregate_datasets.append({'dataset': ds, 'i': metalearner, **ds_dict})
    return pd.DataFrame(aggregate_datasets)


def plot_boxplot(data, prop_name, folder):
    plt.figure(prop_name).suptitle(prop_name)
    sns.boxplot(data=data, y=prop_name)
    plt.savefig(folder)
    plt.close()


def plot_multiple_boxplot(data, prop_name, folder):
    plt.figure(prop_name).suptitle(prop_name)
    g = sns.boxplot(data=data, x='i', y=prop_name, hue='i', dodge=False)
    plt.legend(title='Estrategias', loc='best')
    g.set(xticklabels=[])
    g.set(xlabel=None)
    plt.savefig(folder)
    plt.close()


def plot_histogram(data, metalearners, folder):
    autogoal = metalearners[0]
    fig, axs = plt.subplots(nrows=len(metalearners)-1)
    for i, metalearner in enumerate(metalearners[1:]):
        df = data[(data['i'] == autogoal) | (data['i'] == metalearner)]
        axs[i].set_xlabel('Accuracy Obtenido')
        axs[i].set_ylabel('Datasets')
        sns.histplot(data=df, x='best_fn', hue='i', bins=20, ax=axs[i])
        axs[i].legend(title='Estrategias', loc='best', labels=[autogoal, metalearner])
    plt.savefig(folder / 'histogram')
    plt.close()


def plot_results(metalearners, metalearners_path: Path, plot_folder: Path):
    avg_results = []
    for j, metalearner in enumerate(metalearners):
        metalearner_folder = get_plot_folder(plot_folder / metalearner)
        data = extract_scores(metalearners_path / metalearner)
        avg = build_average_dataframe(data, metalearner)
        data.append(avg)
        avg_results.append(avg)
        for i, df in enumerate(data):
            for column in df.columns:
                if column in ['i', 'dataset']:
                    continue
                plot_boxplot(df, column, metalearner_folder / f'{column}_{i}')
    df = pd.concat(avg_results)
    df.loc[df['best_fn'] < 0, 'best_fn'] = 0
    for column in df.columns:
        if column in ['i', 'dataset']:
            continue
        plot_multiple_boxplot(df, column, plot_folder / f'{column}')

    plot_histogram(df, metalearners, plot_folder)

    dfs = build_performance_info(metalearners, metalearners_path)
    plotting_performance(dfs, plot_folder / 'performance')


def fix_performance_info(performance: list):
    new_performance = [performance[0]]
    for p in performance:
        if p > new_performance[-1]:
            new_performance.append(p)
        else:
            new_performance.append(new_performance[-1])
    return new_performance


def build_performance_info(metalearners, metalearners_path: Path):
    dataframes = {}
    for metalearner in metalearners:
        data_dict = {'i': [], metalearner: []}
        metalearner_path = metalearners_path / metalearner
        for fn in metalearner_path.glob('*.json'):
            info = json.load(open(fn, 'r+'))

            # performance = info['scores']
            #
            performance = [p if p > 0 else 0 for p in info['scores']]
            # performance = [p if p > 0 else 0 for p in info['scores']]
            if len(performance) == 0:
                continue
            performance = fix_performance_info(performance)
            data_dict[metalearner].extend(performance)
            data_dict['i'].extend(range(len(performance)))
        dataframes[metalearner] = pd.DataFrame(data_dict)
    return dataframes


def plotting_performance(data, folder):
    fig = plt.figure()
    for metalearner, df in data.items():
        sns.lineplot(data=df, x='i', y=metalearner)
    # fig.legend(title='Estrategias', labels=data.keys())
    plt.legend(loc='best', title='Estrategias', labels=data.keys())
    plt.ylabel(None)
    plt.xlabel('Iteraciones')
    plt.xlim([0, 200])
    plt.savefig(folder)
    plt.close()


def main():
    # plot_folder = get_plot_folder('plots/results/l1 distance')
    # metalearners = ['autogoal', 'nn_learner_aggregated', 'nn_learner_simple', 'xgb_metalearner']
    # plot_results(metalearners, Path('results/l1 distance/results'), plot_folder)

    # plot_folder = get_plot_folder('plots/results/l2 distance')
    # metalearners = ['autogoal', 'nn_learner_aggregated', 'nn_metalearner_simple', 'xgb_metalearner']
    # plot_results(metalearners, Path('results/l2 distance/results'), plot_folder)

    # plot_folder = get_plot_folder('plots/results/xgb_metalearner_v2')
    # metalearners = ['autogoal', 'xgb_metalearner_v2',
    #                 'nn_learner_aggregated_l1', 'nn_learner_simple_l1', 'xgb_metalearner_l1',
    #                 'nn_learner_aggregated_l2', 'nn_metalearner_simple_l2', 'xgb_metalearner_l2']
    # plot_results(metalearners, Path('results/xgb_metalearner_v2/results'), plot_folder)

    plot_folder = get_plot_folder('plots/results/paper')
    metalearners = ['Autogoal', 'Vecinos Cercanos Simple',
                    'Vecinos Cercanos Ponderado', 'XGBRanker']
    plot_results(metalearners, Path('results/paper/results'), plot_folder)

    # plot_folder = get_plot_folder('plots/results/new_paper')
    # metalearners = ['Autogoal', 'Vecinos Cercanos Simple',
    #                 'Vecinos Cercanos Ponderado', 'XGBRanker']
    # plot_results(metalearners, Path('results/new paperr/results'), plot_folder)

    # plot_folder = get_plot_folder('plots/results/new_paper_performance')
    # metalearners = ['Autogoal', 'Vecinos Cercanos Simple',
    #                 'Vecinos Cercanos Ponderado', 'XGBRanker']
    # plot_performance(metalearners, Path('results/new paperr/results'), plot_folder)


if __name__ == '__main__':
    main()
