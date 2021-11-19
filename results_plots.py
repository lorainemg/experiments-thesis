import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import get_plot_folder
from functools import reduce
from pathlib import Path
from typing import Dict, List
import json
import re


name_for_properties = {
    'failed_pipelines': 'Flujos de Algoritmos Inválidos',
    'best_fn': 'Resultados de Rendimiento',
    'best_fn_normalize': 'Z-Score'
}


def build_info(file_name, info, i, datasets: dict):
    """The datasets results are processed, and stored in a pandas dataframe"""
    data = {key: value for key, value in info.items() if not isinstance(value, list)}
    data['failed_pipelines'] = min(data['failed_pipelines'] / len(info['scores']), 1)
    data['dataset'] = file_name
    data['i'] = i
    if 'max_idx' not in data:
        try:
            max_idx = info['scores'].index(data['best_fn'])
        except:
            max_idx = None
        data['max_idx'] = max_idx
    data['max_idx'] = data['max_idx'] / len(info['scores'])

    # performance = [p if p > 0 else 0 for p in info['scores']]
    median = np.mean(datasets[file_name])
    variance = np.std(datasets[file_name])
    data['best_fn_normalize'] = (data['best_fn'] - median) / variance
    return data


def get_performances(result_folder: Path, datasets_folder: Path, meta_learners: list):
    """Get the list of performance of all the datasets in all the experiments"""
    datasets_performance = {}
    _get_performance(datasets_folder, re.compile('(\d+)\.json'), datasets_performance, 'meta_targets')
    file_re = re.compile('\w+_(\d+)_(\d+)\.json')

    for meta_learner in meta_learners:
        p = result_folder / meta_learner
        _get_performance(p, file_re, datasets_performance, 'scores')

    return datasets_performance


def _get_performance(folder: Path, name_re: re.Pattern, dataset_performance: dict, scores_key: str):
    """Gets the performance of all the dataset of one experiment"""
    for fn in folder.glob('*.json'):
        info = json.load(open(fn, 'r+'))
        m = name_re.match(fn.name)
        if m is None:
            continue
        dataset_name = m.group(1)
        performance = [p if p > 0 else 0 for p in info[scores_key]]
        try:
            dataset_performance[dataset_name].extend(performance)
        except KeyError:
            dataset_performance[dataset_name] = performance
    return dataset_performance


def extract_scores(metalearner_path: Path, datasets):
    """Extracts a feature from an especific task info"""
    data_dict = {}
    file_re = re.compile('\w+_(\d+)_(\d+)\.json')
    for fn in metalearner_path.glob('*.json'):
        info = json.load(open(fn, 'r+'))
        m = file_re.match(fn.name)
        dataset_name = m.group(1)
        iteration = int(m.group(2))

        try:
            data_dict[iteration].append(build_info(dataset_name, info, iteration, datasets))
        except KeyError:
            data_dict[iteration] = [build_info(dataset_name, info, iteration, datasets)]

    return [pd.DataFrame(data) for data in data_dict.values()]


def build_average_dataframe(dfs, metalearner):
    """Builds an average dataframe in case consecutive runs are executed"""
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
    """Plots a boxplot of a property of the data frame and stores the picture in a given folder"""
    plt.figure(prop_name).suptitle(prop_name)
    sns.boxplot(data=data, y=prop_name)
    plt.savefig(f'{folder}.pdf', format='pdf')
    plt.close()


def _get_plot_name(prop_name):
    try:
        return name_for_properties[prop_name]
    except KeyError:
        return prop_name


def plot_multiple_boxplot(data, prop_name, folder):
    """Plots a boxplot of multiple strategies"""
    label_name = _get_plot_name(prop_name)
    plt.figure(prop_name).suptitle(label_name)
    g = sns.boxplot(data=data, x='i', y=prop_name, hue='i', dodge=False) #, showfliers=False)
    plt.legend(title='Estrategias', loc='best')
    g.set(xticklabels=[])
    g.set(xlabel=None)
    g.set(ylabel=label_name)
    plt.ylim([0, 2.7])
    plt.savefig(f'{folder}.pdf', format='pdf')
    plt.close()


def plot_histogram(data, metalearners, folder):
    """Plots a histogram to include different performance of a list of strategies"""
    autogoal = metalearners[0]
    fig, axs = plt.subplots(nrows=len(metalearners)-1)
    for i, metalearner in enumerate(metalearners[1:]):
        df = data[(data['i'] == autogoal) | (data['i'] == metalearner)]
        axs[i].set_xlabel('Accuracy Obtenido')
        axs[i].set_ylabel('Datasets')
        sns.histplot(data=df, x='best_fn', hue='i', bins=20, ax=axs[i])
        axs[i].legend(title='Estrategias', loc='best', labels=[autogoal, metalearner])
    plt.savefig(folder / 'histogram.pdf')
    plt.close()


def plot_results(metalearners, metalearners_path: Path, plot_folder: Path, datasets_folder: Path):
    """
    Main function that plots the results of a set of meta-learners stored in a given path.
    The plots are stored in a given folder, and the folder where the dataset info is stored is also needed
    """
    avg_results = []
    datasets = get_performances(metalearners_path, datasets_folder, metalearners)
    for j, metalearner in enumerate(metalearners):
        metalearner_folder = get_plot_folder(plot_folder / metalearner)
        data = extract_scores(metalearners_path / metalearner, datasets)
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

    # rank_df = build_ranking(metalearners, df)
    # plot_multiple_boxplot(rank_df, 'rank', plot_folder / 'rank')

    for column in df.columns:
        if column in ['i', 'dataset']:
            continue
        plot_multiple_boxplot(df, column, plot_folder / f'{column}')

    plot_histogram(df, metalearners, plot_folder)

    dfs = build_performance_info(metalearners, metalearners_path, normalize=False)
    plotting_performance(dfs, plot_folder / 'performance')

    # dfs = build_average_ranking(dfs)
    # plotting_performance(dfs, plot_folder / 'ranking')

    dfs = build_performance_info(metalearners, metalearners_path, normalize=True)
    plotting_performance(dfs, plot_folder / 'normalize_performance')


def normalize_dataset(performance: list, variance: float, median: float):
    """Normalizes the performance obtain in a given dataset"""
    return [(p - median) / variance for p in performance]


def fix_performance_info(performance: list):
    new_performance = [performance[0]]
    for p in performance[1:]:
        if p > new_performance[-1]:
            new_performance.append(p)
        else:
            new_performance.append(new_performance[-1])
    return new_performance


def build_performance_info(metalearners, metalearners_path: Path, normalize=False):
    """Extracts the performance info of the given strategies, and a pandas dataframe is built"""
    dataframes = {}
    for metalearner in metalearners:
        data_dict = {'i': [], metalearner: []}
        metalearner_path = metalearners_path / metalearner
        for fn in metalearner_path.glob('*.json'):
            info = json.load(open(fn, 'r+'))

            # performance = info['scores']
            #

            performance = [p if p > 0 else 0 for p in info['scores']]
            if len(performance) == 0:
                continue
            if normalize:
                median = np.mean(performance)
                var = np.var(performance)
                performance = normalize_dataset(performance, var, median)
            performance = fix_performance_info(performance)

            data_dict[metalearner].extend(performance)
            data_dict['i'].extend(range(len(performance)))
        dataframes[metalearner] = pd.DataFrame(data_dict)
    return dataframes


def build_average_ranking(dataframes: Dict[str, pd.DataFrame]):
    """Builds a ranking with the dataframe performance, a dataframe is returned with this rankings"""
    dataframes_rank = {}
    data_dict = {metalearner: {'i': [], metalearner: []} for metalearner in dataframes.keys()}
    for idx in range(200):
        results = {}
        datasets = 0
        for metalearner in dataframes.keys():
            dataframe = dataframes[metalearner]
            results[metalearner] = list(dataframe[dataframe['i'] == idx][metalearner])
            datasets = max(datasets, len(results[metalearner]))
        ranking = {metalearner: [] for metalearner in dataframes.keys()}
        for i in range(datasets):
            ranks = {}
            for metalearner in results.keys():
                try:
                    performance = results[metalearner][i]
                except IndexError:
                    continue
                ranks[metalearner] = performance
            mtl_rank = [mtl for mtl, _ in sorted(ranks.items(), key=lambda x: x[1], reverse=True)]

            previous_performance = 0
            for j, mtl in enumerate(mtl_rank):
                if j > 0 and ranks[mtl] == previous_performance:
                    ranking[mtl].append(ranking[mtl_rank[j - 1]][-1])
                else:
                    ranking[mtl].append(j)
                previous_performance = ranks[mtl]
        for mtl, rank in ranking.items():
            data_dict[mtl]['i'].extend([idx]*len(rank))
            data_dict[mtl][mtl].extend(rank)
    return {metalearner: pd.DataFrame(data) for metalearner, data in data_dict.items()}


def build_ranking(metalearners: List[str], data: pd.DataFrame):
    """Builds a ranking with the best performance info"""
    datasets = len(data[data['i'] == 'autogoal']['best_fn'])
    ranking = {metalearner: [] for metalearner in metalearners}
    for d in range(datasets):
        performance = {}
        for mtl in metalearners:
            fn = data[data['i'] == mtl]['best_fn'][d]
            performance[mtl] = fn
        mtl_rank = [mtl for mtl, _ in sorted(performance.items(), key=lambda x: x[1], reverse=True)]
        prev_performance = 0
        for i, mtl in enumerate(mtl_rank):
            if i > 0 and performance[mtl] == prev_performance:
                ranking[mtl].append(ranking[mtl_rank[i - i]][-1])
            else:
                ranking[mtl].append(i)
            prev_performance = performance[mtl]
    data = {'i': [], 'rank': []}
    for mtl, rank in ranking.items():
        data['i'].extend([mtl]*len(rank))
        data['rank'].extend(rank)
    return pd.DataFrame(data)


def plotting_performance(data, folder):
    """
    Helper function that plots the performance of the given strategies
    results are stored in the given folder
    """
    fig = plt.figure()
    for metalearner, df in data.items():
        sns.lineplot(data=df, x='i', y=metalearner)
    # fig.legend(title='Estrategias', labels=data.keys())
    plt.legend(loc='best', title='Estrategias', labels=data.keys())
    plt.ylabel(None)
    plt.xlabel('Iteraciones')
    plt.xlim([0, 200])
    plt.ylabel('Resultados de Rendimiento')
    plt.savefig(f'{folder}.pdf')
    plt.close()


def main():
    plot_folder = get_plot_folder('plots/results/server')
    metalearners = ['AutoGOAL', 'Vecinos Cercanos Simple',
                    'Vecinos Cercanos Ponderado', 'XGBRanker']
    plot_results(metalearners, Path('results/server/results'), plot_folder, Path('results/datasets_info/Classification'))


if __name__ == '__main__':
    main()
