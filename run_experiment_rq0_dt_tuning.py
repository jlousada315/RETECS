#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Experiment, evaluation and visualization for RQ0
from run_experiment_common import *

PARALLEL = False
RUN_EXPERIMENT = False
VISUALIZE_RESULTS = True
DATA_DIR = 'RESULTS'
PARALLEL_POOL_SIZE = 2

max_depths = [2, 4, 8, 20]
min_samples_split = [2, 3, 5, 10]
criterion = ['gini', 'entropy']


def exp_tree_params(iteration):
    reward_function = reward.tcfail
    avg_napfd = []
    for c in criterion:
        for d in max_depths:
            for s in min_samples_split:
                agent = agents.DTAgent(histlen=retecs.DEFAULT_HISTORY_LENGTH,
                                       action_size=1,
                                       criterion=c,
                                       max_depth=d,
                                       min_samples_split=s)

                scenario = get_scenario('bnp')

                file_appendix = 'rq0_%s_criterion%s_depth%s_min_samples%s_%d' % (
                    agent.name, c, d, s, iteration)

                rl_learning = retecs.PrioLearning(agent=agent,
                                                  scenario_provider=scenario,
                                                  reward_function=reward_function,
                                                  preprocess_function=retecs.preprocess_discrete,
                                                  file_prefix=file_appendix,
                                                  dump_interval=100,
                                                  validation_interval=0,
                                                  output_dir=DATA_DIR)

                res = rl_learning.train(no_scenarios=CI_CYCLES,
                                        print_log=False,
                                        plot_graphs=False,
                                        save_graphs=False,
                                        collect_comparison=False)
                avg_napfd.append(res)
    return avg_napfd


def visualize():
    search_pattern = 'rq0_*_criterion*_depth*_min_samples*_*_stats.p'
    filename = 'rq0_depth_min_sample'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)
    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    df = df[~df['agent'].isin(['heur_random', 'heur_sort', 'heur_weight'])]

    rel_df = df.groupby(['agent', 'criterion', 'depth', 'min_samples_split'], as_index=False).mean()
    rel_df['napfd'] = rel_df['napfd'] / max(rel_df['napfd']) * 100

    fig, axes = plt.subplots(2, 1, sharey=True)
    sns.set(font_scale=1.4)

    ax = sns.barplot(x='depth', y='napfd', hue='min_samples_split',
                     data=rel_df[rel_df['criterion'] == 'gini'], ax=axes[0])
    ax1 = sns.barplot(x='depth', y='napfd', hue='min_samples_split',
                      data=rel_df[rel_df['criterion'] == 'entropy'], ax=axes[1])

    ax.set_xlabel('')
    ax1.set_xlabel('')
    ax.set_ylabel('')
    ax1.set_ylabel('')

    ax.set_title('(a) Gini')
    ax1.set_title('(b)) Entropy')
    fig.text(0.5, 0.04, 'Depth', ha='center')
    fig.text(0.04, 0.5, '% of best result', va='center', rotation='vertical')

    ax.grid(zorder=0)
    ax1.grid(zorder=0)
    plt.locator_params(axis='y', nbins=5)

    handles, labels = ax1.get_legend_handles_labels()
    ax.get_legend().remove()
    sns.set(font_scale=1)
    plt.legend(handles=handles, loc="lower left", bbox_to_anchor=[0.45, 0.05],
               ncol=4, shadow=True, title="Min_Samples_Split", fancybox=True, prop={'size': 8})

    #fig.suptitle('Decision Tree Parameter Tuning')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    #save_figures(fig, filename)
    plt.savefig('Images/' + filename + '.eps', format='eps')

    fig.tight_layout()


if __name__ == '__main__':
    if RUN_EXPERIMENT:
        run_experiments(exp_tree_params, parallel=PARALLEL)

    if VISUALIZE_RESULTS:
        visualize()
