"""
    @author María Andrea Cruz Blandón
    @date 02.11.2023

    This script calculates the statistics and effect size for tone discrimination.
"""
import math

import pandas as pd
from pathlib import Path

__docformat__ = ['reStructuredText']
__all__ = ['obtain_effect_size']

from scipy.stats import ttest_ind, norm


def _get_effects_and_statistics(group: pd.DataFrame) -> dict:
    same_data = group[group['condition'] == 'same']['distance']
    different_data = group[group['condition'] == 'different']['distance']

    d = (different_data.mean() - same_data.mean()) / math.sqrt((same_data.std() ** 2 + different_data.std() ** 2) / 2)
    n1 = len(same_data)
    n2 = len(different_data)
    g = d * (1 - 3 / (4 * (n1 + n2 - 2) - 1))
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + (d ** 2 / (2 * n1 * n2)))
    se_g = math.sqrt((n1 + n2) / (n1 * n2) + (g ** 2 / (2 * n1 * n2)))

    # 95% CI
    z = norm.ppf(0.975)  # alpha 0.05
    ci_lb_d = d - z * math.sqrt(se_d)
    ci_ub_d = d + z * math.sqrt(se_d)
    ci_lb_g = g - z * math.sqrt(se_g)
    ci_ub_g = g + z * math.sqrt(se_g)

    t_statistic, p_value = ttest_ind(same_data, different_data, equal_var=False)

    return {'t': t_statistic, 'p_value': p_value, 'd': d, 'g': g, 'se_d': se_d, 'se_g': se_g,
                      'ci_lb_d': ci_lb_d, 'ci_ub_d': ci_ub_d, 'ci_lb_g': ci_lb_g, 'ci_ub_g': ci_ub_g}


def obtain_effect_size(distances_path: Path, output_path: Path, test_type: str) -> None:
    df = pd.read_csv(distances_path)
    # calculate t-test statistics and p-value
    if test_type == 'basic':
        df = df[df['syllable'] == 'ci']

    effects = _get_effects_and_statistics(df)
    effects['type'] = 'tone_discrimination'

    df_effects = pd.DataFrame([effects])
    df_effects = df_effects[['type', 'd', 'g', 't', 'se_d', 'se_g', 'ci_lb_d', 'ci_ub_d', 'ci_lb_g', 'ci_ub_g', 'p_value']]

    output_path.mkdir(parents=True, exist_ok=True)
    df_effects.to_csv(output_path.joinpath('output.csv'), index=False)
