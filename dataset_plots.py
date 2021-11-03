import json
from pathlib import Path
from utils import get_plot_folder

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_info(file_name, meta_features):
    """Builds the info of a dataset in a presentable manner"""
    info = {'dataset': int(file_name[:-5])}
    for feat_name, value in meta_features.items():
        new_name = feat_name.replace("_", " ")
        info[new_name] = value
    return info


def extract_feature(task):
    """Extracts a feature from an specific task info"""
    info_folder = Path('results/95_datasets_info') / task
    data_dict = []
    for fn in info_folder.glob('*.json'):
        info = json.load(open(fn, 'r+'))
        data_dict.append(build_info(fn.name, info['meta_features']))
    return pd.DataFrame(data_dict)


def plot_numerical_values(data, prop_name):
    """Plots numerical values in a kde format"""
    #     sns.displot(data=data[prop_name], kind='kde')
    plt.figure(prop_name)
    sns.kdeplot(x=prop_name, data=data, shade=True)


def plot_categorical_values(data, prop_name):
    """Plots categorical values counting values"""
    plt.figure(prop_name)
    sns.countplot(x=prop_name, data=data, palette='rainbow')


def plot_values(data, values, plot_folder, type_):
    """Plots categorical or numeric values"""
    for value in values:
        if type_ == 'numerical':
            plot_numerical_values(data, value)
        elif type_ == 'categorical':
            try:
                plot_categorical_values(data, value)
            except:
                print(value)
        plt.savefig(plot_folder / f'{value}.pdf')
        plt.close()


def plot_scatterplot(data, plot_folder):
    """Scatter plot of #instances-#features, #instances-#classes (auto-sklearn style)"""
    fig, axs = plt.subplots(nrows=2)
    sns.scatterplot(data=data, x='input dimensionality', y='number of samples', ax=axs[0])
    sns.scatterplot(data=data, x='input dimensionality', y='number of classes', ax=axs[1])
    plt.savefig(plot_folder / f'scatterplot.pdf')
    plt.close()


def plot_lineplot(data, plot_folder):
    """Line plot of #instances, #features, #classes - dataset id (atm style)"""
    fig, axs = plt.subplots(nrows=3)
    datasets_number = range(len(data))
    sns.lineplot(data=data, x=datasets_number, y='number of samples', ax=axs[0])
    sns.lineplot(data=data, x=datasets_number, y='input dimensionality', ax=axs[1])
    sns.lineplot(data=data, x=datasets_number, y='number of classes', ax=axs[2])
    plt.savefig(plot_folder / f'lineplot.pdf')
    plt.close()


def main():
    """Does all main stuff"""
    numerical_features = ['number of samples', 'input dimensionality',
                          'output dimensionality', 'dataset dimensionality', 'standard deviation',
                          'coefficient of variation', 'covariance avg', 'linear corr coef',
                          'skewness', 'skewness 1', 'skewness 2', 'skewness 3', 'kurtosis',
                          'kurtosis 1', 'kurtosis 2', 'kurtosis 3', 'normalized class entropy',
                          'normalized attr entropy', 'normalized attr entropy 1',
                          'normalized attr entropy 2', 'normalized attr entropy 3',
                          'joint entropy', 'joint entropy 1', 'joint entropy 2',
                          'joint entropy 3', 'mutual information', 'equivalent number of attr',
                          'noise signal ratio']
    categorical_features = ['is supervised', 'has numeric features',
                            'average number of words', 'has text features', 'semantic input types',
                            'semantic output types']

    df = extract_feature('Classification')  # build dataframe with all datasets characteristics

    plt_folder = get_plot_folder('plots/meta_features')

    plot_scatterplot(df, plt_folder)
    plot_lineplot(df, plt_folder)

    # Plots numerical values
    plot_values(df, numerical_features, plt_folder, 'numerical')

    # Plots categorical values
    plot_values(df, categorical_features, plt_folder, 'categorical')


if __name__ == '__main__':
    main()
