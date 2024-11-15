"""
    @author María Andrea Cruz Blandón
    @date 14.10.2023

    This script calculates the statistics and effect size per contrast and for the whole test.
"""
import math

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union

__docformat__ = ['reStructuredText']
__all__ = ['obtain_effect_size']

from scipy.stats import ttest_ind, norm


def _get_effects_and_statistics(group: pd.DataFrame) -> pd.Series:
    same_data = group[group['condition'] == 'same']['distance']
    different_data = group[group['condition'] == 'different']['distance']

    d = (different_data.mean() - same_data.mean()) / math.sqrt((same_data.std() ** 2 + different_data.std() ** 2) / 2)
    n1 = len(same_data)
    n2 = len(different_data)
    g = d * (1 - 3 / (4 * (n1 + n2 - 2) - 1))
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + (d ** 2 / (2 * n1 * n2)))
    se_g = math.sqrt((n1 + n2) / (n1 * n2) + (g ** 2 / (2 * n1 * n2)))
    w_d = 1 / (se_d ** 2)
    w_g = 1 / (se_g ** 2)
    t_statistic, p_value = ttest_ind(same_data, different_data, equal_var=False)

    return pd.Series({'t_statistic': t_statistic, 'p_value': p_value, 'd': d, 'g': g, 'se_d': se_d, 'se_g': se_g,
                      'w_d': w_d, 'w_g': w_g, 'n1': n1, 'n2': n2})


def _get_overall_effect(group: pd.DataFrame) -> pd.Series:
    group['weight_d'] = group['w_d'] * group['d']
    group['weight_g'] = group['w_g'] * group['g']
    mean_es_d = group['weight_d'].sum() / group['w_d'].sum()
    mean_es_g = group['weight_g'].sum() / group['w_g'].sum()
    se_mean_d = math.sqrt(1 / group['w_d'].sum())
    se_mean_g = math.sqrt(1 / group['w_g'].sum())
    # confidence interval 95%
    z = norm.ppf(.975)  # alpha = 0.05
    ci_lb_d = mean_es_d - (z * se_mean_d)
    ci_ub_d = mean_es_d + (z * se_mean_d)
    ci_lb_g = mean_es_g - (z * se_mean_g)
    ci_ub_g = mean_es_g + (z * se_mean_g)
    # determine p-value
    p_value = norm.cdf(-math.fabs(mean_es_d)/se_mean_d) * 2

    return pd.Series({'d': mean_es_d, 'g': mean_es_g, 'se_d': se_mean_d, 'se_g': se_mean_g,
                      'ci_lb_d': ci_lb_d, 'ci_ub_d': ci_ub_d, 'ci_lb_g': ci_lb_g, 'ci_ub_g': ci_ub_g,
                      'p_value': p_value})


def obtain_effect_size(distances_path: Path, output_path: Path) -> None:
    df = pd.read_csv(distances_path)
    # calculate t-test statistics and p-value per contrast, langauge, and corpus
    groups = df.groupby(['contrast', 'language', 'corpus', 'type'])
    effects = groups.apply(_get_effects_and_statistics).reset_index()

    output_path.mkdir(parents=True, exist_ok=True)

    effects.to_csv(output_path.joinpath('effects_per_contrast.csv'), index=False)

    # get overall effect
    groups_type = effects.groupby(['type'])
    overall_effect = groups_type.apply(_get_overall_effect).reset_index()
    overall_effect['t'] = None
    overall_effect = overall_effect[['type', 'd', 'g', 't', 'se_d', 'se_g', 'ci_lb_d', 'ci_ub_d', 'ci_lb_g', 'ci_ub_g',
                                     'p_value']]

    overall_effect.to_csv(output_path.joinpath('output.csv'), index=False)


