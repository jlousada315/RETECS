#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ1
import matplotlib

from run_experiment_common import *

# For overriding defaults from run_experiment_common
PARALLEL = False
RUN_EXPERIMENT = False
VISUALIZE_RESULTS = True


def visualize2():
    matplotlib.rcParams.update({'font.size': 30})

    search_pattern = 'rq_*_stats.p'
    filename = 'rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)
    df = stats.load_stats_dataframe(iteration_results, aggregated_results)

    pure_df = df[(~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])) & (df['detected'] + df['missed']) > 0]
    mean_df = pure_df.groupby(['step', 'env', 'agent', 'rewardfun'], as_index=False).mean()

    env = 'bnp'
    rewardfun = 'tcfail'
    agent = 'mlpclassifier'

    rel_df = mean_df[(mean_df['env'] == env) & (mean_df['rewardfun'] == rewardfun)]
    rel_df[rel_df['agent'] == agent].plot(x='step', y='napfd', ylim=[0, 1], linewidth=1.5, label=None)
    x = rel_df.loc[rel_df['agent'] == agent, 'step']
    y = rel_df.loc[rel_df['agent'] == agent, 'napfd']

    trend = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(x, trend(x), color='k', linewidth=1.5, label=None)
    plt.ylabel('APFD')
    plt.xlabel('Commits')
    plt.legend('', frameon=False)


    plotname = 'single_apfd'


    """
    # One groupplot
    fig, axarr = plt.subplots(3, 3, sharey=True, sharex='col', figsize=figsize_text(1.2, 1.4))
    subplot_labels = ['(a)', '(b)', '(c)']

    for column, env in enumerate(sorted(mean_df['env'].unique(), reverse=True)):
        print(env)
        for row, rewardfun in enumerate(mean_df['rewardfun'].unique()):
            for agidx, (labeltext, agent, linestyle) in enumerate(
                    [('Network', 'mlpclassifier', '-'), ('DecisionTree', 'dtclassifier', '-.')]):
                rel_df = mean_df[(mean_df['env'] == env) & (mean_df['rewardfun'] == rewardfun)]
                rel_df[rel_df['agent'] == agent].plot(x='step', y='napfd', label=labeltext, ylim=[0, 1], linewidth=0.8,
                                                      style=linestyle, color=sns.color_palette()[agidx],
                                                      ax=axarr[row, column])

                x = rel_df.loc[rel_df['agent'] == agent, 'step']
                y = rel_df.loc[rel_df['agent'] == agent, 'napfd']

                trend = np.poly1d(np.polyfit(x, y, 1))
                axarr[row, column].plot(x, trend(x), linestyle, color='k', linewidth=0.8)

            axarr[row, column].legend_.remove()

            axarr[row, column].set_xticks(np.arange(0, 360, 60), minor=False)
            axarr[row, column].set_xticklabels([0, 60, 120, 180, 240, 300], minor=False)

            # axarr[row, column].xaxis.grid(True, which='minor')

            if column == 1:
                axarr[row, column].set_title('%s %s' % (subplot_labels[row], reward_names[rewardfun]))

            if row == 0:
                if column == 1:
                    axarr[row, column].set_title(
                        '%s\n%s %s' % (env_names[env], subplot_labels[row], reward_names[rewardfun]))
                else:
                    axarr[row, column].set_title(env_names[env] + '\n')
            elif row == 2:
                axarr[row, column].set_xlabel('CI Cycle')

            if column == 0:
                axarr[row, column].set_ylabel('NAPFD')

            if row == 0 and column == 2:
                axarr[row, column].legend(loc='upper right', ncol=2, frameon=True)
    """
    plt.tight_layout()
    # save_figures(fig, plotname)
    plt.savefig(plotname + '.eps', format='eps')

    plt.clf()


def visualize():
    search_pattern = 'rq_*_stats.p'
    filename = 'rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)
    df = stats.load_stats_dataframe(iteration_results, aggregated_results)

    pure_df = df[(~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])) & (df['detected'] + df['missed']) > 0]
    mean_df = pure_df.groupby(['step', 'env', 'agent', 'rewardfun'], as_index=False).mean()

    # One groupplot
    fig, axarr = plt.subplots(3, 3, sharey=True, sharex='col', figsize=figsize_text(1.2, 1.4))
    plotname = 'rq1_napfd'
    subplot_labels = ['(a)', '(b)', '(c)']

    for column, env in enumerate(sorted(mean_df['env'].unique(), reverse=True)):
        print(env)
        for row, rewardfun in enumerate(mean_df['rewardfun'].unique()):
            for agidx, (labeltext, agent, linestyle) in enumerate(
                    [('Network', 'mlpclassifier', '-'), ('DecisionTree', 'dtclassifier', '-.')]):
                rel_df = mean_df[(mean_df['env'] == env) & (mean_df['rewardfun'] == rewardfun)]
                rel_df[rel_df['agent'] == agent].plot(x='step', y='napfd', label=labeltext, ylim=[0, 1], linewidth=0.8,
                                                      style=linestyle, color=sns.color_palette()[agidx],
                                                      ax=axarr[row, column])

                x = rel_df.loc[rel_df['agent'] == agent, 'step']
                y = rel_df.loc[rel_df['agent'] == agent, 'napfd']

                trend = np.poly1d(np.polyfit(x, y, 1))
                axarr[row, column].plot(x, trend(x), linestyle, color='k', linewidth=0.8)

            axarr[row, column].legend_.remove()

            axarr[row, column].set_xticks(np.arange(0, 360, 60), minor=False)
            axarr[row, column].set_xticklabels([0, 60, 120, 180, 240, 300], minor=False)

            # axarr[row, column].xaxis.grid(True, which='minor')

            if column == 1:
                axarr[row, column].set_title('%s %s' % (subplot_labels[row], reward_names[rewardfun]))

            if row == 0:
                if column == 1:
                    axarr[row, column].set_title(
                        '%s\n%s %s' % (env_names[env], subplot_labels[row], reward_names[rewardfun]))
                else:
                    axarr[row, column].set_title(env_names[env] + '\n')
            elif row == 2:
                axarr[row, column].set_xlabel('CI Cycle')

            if column == 0:
                axarr[row, column].set_ylabel('NAPFD')

            if row == 0 and column == 2:
                axarr[row, column].legend(loc='upper right', ncol=2, frameon=True)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    # save_figures(fig, plotname)
    plt.savefig(plotname + '.eps', format='eps')

    plt.clf()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        run_experiments(exp_run_industrial_datasets, parallel=PARALLEL)

    if VISUALIZE_RESULTS:
        visualize2()
