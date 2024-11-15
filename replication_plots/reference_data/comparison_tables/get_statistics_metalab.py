"""
    @author María Andrea Cruz Blandón
    @date 20.11.2023

    Produce the compatibility tables as well as reference table for the linguistic capabilities and different
    simulations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme()


def get_reference_data_table(reference_csv: Path) -> None:
    df_data = pd.read_csv(reference_csv, keep_default_na=False)

    df_tmp = df_data.loc[:, df_data.columns != 'Capability']

    df_np = df_tmp.to_numpy()

    # Formatting & data
    # Basic format
    result_type = np.zeros((df_np.shape[0], df_np.shape[1]))
    annotations = np.copy(df_np)

    n_cap, n_checkpoints = df_np.shape
    caps = {0: 'IDS Preference', 1: 'Vowel discrimination',
            2: 'Tone discrimination (full-term)', 3: 'Tone discrimination (preterm)',
            4: 'Phonotactics preference (full-term)', 5: 'Phonotactics preference (preterm)'}
    anno = {'yes': '✔', 'no': '✘'}

    for i in range(n_cap):
        for j in range(n_checkpoints):
            es_inf = df_np[i, j]
            cap = caps[i]
            annotations[i, j] = f'{anno[es_inf]}' if es_inf in ['yes', 'no'] else f'{float(es_inf):.2f}'

            if es_inf == 'yes':
                result_type[i, j] = 1  # Compatible effect
            elif es_inf == 'no':
                result_type[i, j] = 2  # Non compatible effect
            else:
                if float(es_inf) > 0:
                    result_type[i, j] = 1
                else:
                    result_type[i, j] = 2

    # Plot
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 13), gridspec_kw={'width_ratios': [1, 0.05]})
    title = "Infant reference data\n"

    ax.set_title(title, fontsize=24, fontweight='bold')

    colours = ['#fdebde', '#9EC3D8', '#edc1bb', '#f7f5f4']
    sns.heatmap(result_type, linewidths=0.5, annot=annotations, fmt='',
                cmap=sns.color_palette(colours, as_cmap=True),
                vmax=3,
                vmin=0,
                ax=ax,
                cbar=False,
                annot_kws={"fontsize": 20})

    sns.heatmap(np.array([1, 2]).reshape(2, 1),
                annot=np.column_stack(['Effect', 'No effect']).reshape(2, 1),
                fmt='', cmap=sns.color_palette(colours[0:3], as_cmap=True),
                vmax=2, vmin=0, ax=ax2, cbar=False, annot_kws={"fontsize": 20, "rotation": 90, 'fontstyle': 'italic'},
                xticklabels=False, yticklabels=False)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(df_tmp.columns.to_list(), rotation=0, ha='right', size=20)
    ax.set_yticks(ax.get_yticks())
    caps = df_data.loc[:, 'Capability'].to_list()
    ax.set_yticklabels(caps, size=20, rotation=0)

    ax.set_xlabel("\nInfant Age (Months)", fontsize=20, fontweight='bold')

    fig.tight_layout()
    plt.savefig('summary_reference_data.pdf')


def get_simulations_tables(simulations_csv: Path, reference_es_csv: Path, reference_lb_csv: Path) -> None:
    df_lb = pd.read_csv(reference_lb_csv, keep_default_na=False)
    df_simulations = pd.read_csv(simulations_csv, keep_default_na=False)
    df_es = pd.read_csv(reference_es_csv, keep_default_na=False)

    # Set reference data
    df_lb_tmp = df_lb.loc[:, df_lb.columns != 'Capability']
    df_es_tmp = df_es.loc[:, df_es.columns != 'Capability']

    age = [float(x) for x in df_es_tmp.columns.to_list()]

    df_lb_np = df_lb_tmp.to_numpy()
    df_es_np = df_es_tmp.to_numpy()

    # Formatting & data
    # Basic format
    anno = {'yes': '✔', 'no': '✘'}
    total_caps = 4  # simulations are either full-term or preterm
    titles = {'full_term': 'Full-term simulation', 'preterm': 'Preterm simulation',
              'baseline': 'Baseline simulation'}
    caps = {0: 'IDS preference', 1: 'Vowel discrimination',
            2: 'Tone discrimination', 3: 'Phonotactics preference'}

    for simulation in ['full_term', 'preterm', 'baseline']:
        # Get simulation data
        df_sim_tmp = df_simulations.loc[df_simulations.simulation == simulation]
        df_sim_tmp = df_sim_tmp.loc[:, df_sim_tmp.columns != 'simulation']

        # initiate result datastructures

        result_type = np.zeros((total_caps, df_lb_np.shape[1]))

        if simulation == 'preterm':
            # remove rows of full-term
            df_lb_np_tmp = np.delete(df_lb_np, [2, 4], 0)
            df_es_np_tmp = np.delete(df_es_np, [2, 4], 0)
            annotations = np.copy(df_lb_np_tmp)
            annotations_inf = np.copy(df_lb_np_tmp)

        else:
            # remove rows of preterm
            df_lb_np_tmp = np.delete(df_lb_np, [3, 5], 0)
            df_es_np_tmp = np.delete(df_es_np, [3, 5], 0)
            annotations = np.copy(df_lb_np_tmp)
            annotations_inf = np.copy(df_lb_np_tmp)

        for model in ['apc', 'cpc']:
            df_sim_tmp2 = df_sim_tmp.loc[df_sim_tmp.model_type == model]
            df_sim_tmp2 = df_sim_tmp2.loc[:, df_sim_tmp2.columns != 'model_type']

            # build table
            n_caps, n_chkpts = df_es_np_tmp.shape

            for i in range(n_caps):
                for j in range(n_chkpts):
                    es_lb = df_lb_np_tmp[i, j]
                    es = df_es_np_tmp[i, j]
                    cap = caps[i]
                    es_sim = float(df_sim_tmp2.loc[df_sim_tmp2.Capability == cap].iat[j, 0])
                    es_sim_sig = df_sim_tmp2.loc[df_sim_tmp2.Capability == cap].iat[j, 1]
                    significance = '*' if es_sim_sig == 'significant' else ''
                    if es_sim_sig == 'not significant':
                        es_sim = 0
                    if i in [0, 1]:
                        annotations[i, j] = (f'{es_sim: .2f}'
                                             f'{significance}') if es_sim_sig == 'significant' else f'{es_sim}'
                        annotations_inf[i, j] = f'\n\n({es_lb})' if es_lb in ['N/A',
                                                                              'n.s.'] else f'\n\n({float(es_lb):.2f})'
                    else:  # tone and phonotactics
                        annotations[i, j] = f'{anno["yes"]}' \
                            if es_sim_sig == 'significant' and es_sim != 0 else f'{anno["no"]}'
                        annotations_inf[i, j] = ''

                    # Compatibility
                    if es_lb == 'n.s.':
                        if es_sim == 0:
                            result_type[i, j] = 1  # Compatible effect
                        else:
                            result_type[i, j] = 2  # Non compatible effect
                    else:
                        if i in [0, 1]:
                            es = float(es)
                            es_lb = float(es_lb)

                            # round to 2 decimals
                            es = round(es, 2)
                            es_lb = round(es_lb, 2)
                            es_sim = round(es_sim, 2)

                            if es > 0:
                                if es_sim >= es_lb:
                                    result_type[i, j] = 1
                                else:
                                    result_type[i, j] = 2
                            elif es < 0:
                                if es_sim <= es_lb:
                                    result_type[i, j] = 1
                                else:
                                    result_type[i, j] = 2
                            else:
                                if es_sim == es_lb:
                                    result_type[i, j] = 1
                                else:
                                    result_type[i, j] = 2
                        else:
                            if (es_sim != 0 and es == 'yes') or (es_sim == 0 and es == 'no'):
                                result_type[i, j] = 1
                            else:
                                result_type[i, j] = 2
            # Plot
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 13), gridspec_kw={'width_ratios': [1, 0.05]})
            title = f"{titles[simulation]}\n{model.upper()}\n"
            ax.set_title(title, fontsize=24, fontweight='bold')
            colours = ['#fdebde', '#9EC3D8', '#edc1bb', '#f7f5f4']
            sns.heatmap(result_type, linewidths=0.5, annot=annotations, fmt='',
                        cmap=sns.color_palette(colours, as_cmap=True),
                        vmax=3,
                        vmin=0,
                        ax=ax,
                        cbar=False,
                        annot_kws={"fontsize": 20})
            sns.heatmap(result_type, linewidths=0.5, annot=annotations_inf, fmt='',
                        cmap=sns.color_palette(colours, as_cmap=True),
                        vmax=3,
                        vmin=0,
                        ax=ax,
                        cbar=False,
                        annot_kws={"fontsize": 20, 'fontstyle': 'italic'})
            sns.heatmap(np.array([1, 2]).reshape(2, 1),
                        annot=np.column_stack(
                            ['Compatible', 'Non compatible']).reshape(2, 1),
                        fmt='', cmap=sns.color_palette(colours[0:3], as_cmap=True),
                        vmax=2, vmin=0, ax=ax2, cbar=False,
                        annot_kws={"fontsize": 20, "rotation": 90, 'fontstyle': 'italic'},
                        xticklabels=False, yticklabels=False)

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(df_es_tmp.columns.to_list(), rotation=0, ha='right', size=20)
            ax.set_yticks(ax.get_yticks())
            caps = [caps[i] for i in range(total_caps)]
            ax.set_yticklabels(caps, size=20, rotation=0)
            ax.set_xlabel("\nSimulated Age (Months)", fontsize=20, fontweight='bold')
            fig.tight_layout()
            plt.savefig(f'summary_table_{simulation}_{model}.pdf')


get_reference_data_table('./summary_reference_data_lb.csv')
get_simulations_tables('./summary_simulations.csv', './summary_reference_data_es.csv', './summary_reference_data_lb.csv')
