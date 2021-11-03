import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import get_plot_folder


def build_info(scores):
    return [{'dataset': dataset, **dscores} for dataset, dscores in scores.items()]


def extract_scores(strategy):
    """Extracts a feature from an specific task info"""
    info_folder = Path('results') / strategy / 'results'
    data_dict = []
    for fn in info_folder.glob('*.json'):
        info = json.load(open(fn, 'r+'))
        data_dict.append(build_info(info))
    return [pd.DataFrame(data) for data in data_dict]


def get_globals(data):
    """Get the globals of different iterations"""
    return pd.DataFrame([df.iloc[-1] for df in data])


def plot_boxplot(data: pd.DataFrame, metric: str, fig_path: Path):
    """Plots the data in a given metric and stores it in the figure path"""
    plt.figure(metric)
    sns.boxplot(data=data, y=metric)
    plt.savefig(fig_path)
    plt.close()


def plot_results(strategies: List[str], metrics: List[str], plot_folder: Path):
    """
    Plots the results of a list of strategies by a given metrics
    and results are stored in plot folder.
    """
    for strategy in strategies:
        data = extract_scores(strategy)
        plot_folder = get_plot_folder(plot_folder / strategy)
        globl = get_globals(data)
        for metric in metrics:
            for i, df in enumerate(data, 1):
                plot_boxplot(df, metric, plot_folder / f'{metric}_{i}.pdf')
            plot_boxplot(globl, metric, plot_folder / f'global_{metric}.pdf')


def main():
    """Configures everything to save all plots"""
    strategies = ['xgb_metalearner']    #, 'nn_metalearner']
    metrics = ['srcc_score', 'wrc_score', 'dcg_score', 'ndcg_score']
    plot_folder = get_plot_folder('plots/meta_learners')
    plot_results(strategies, metrics, plot_folder)


if __name__ == '__main__':
    main()
