"""
    @author María Andrea Cruz Blandón
    @date 07.11.2023

    This script calculates the phonotactics preference (high probable vs low probable triphones in English).
    The coding in this test is such that it is expected to have greater preference for the high probable triphones.

    The attentional preference scores should be given per frame per file.
    This script produces two outputs, an intermediate csv file gather the information of all files in the test,
    and the final csv file with the effect size and its statistics.
"""

import argparse
import math
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind, norm

__docformat__ = ['reStructuredText']
__all__ = ['calculate_phonotactics_preference_d', 'calculate_effect_size']


def _create_scores_file(preferences_path: Path, output_folder: Path) -> Path:
    """
        It creates the intermediate csv file with the information of all files in the test.
    """
    assert preferences_path.exists(), f'The path {preferences_path} does not exist'
    assert preferences_path.joinpath('HP').exists(), f'The path {preferences_path.joinpath("HP")} does not exist'
    assert preferences_path.joinpath('LP').exists(), f'The path {preferences_path.joinpath("LP")} does not exist'

    files = preferences_path.rglob('**/*.npy')

    df = pd.DataFrame(columns=['file_name', 'trial_type', 'frame', 'attentional_preference_score'])

    for score_file in files:
        trial_type = score_file.parent.stem
        scores = np.load(score_file).reshape(-1)
        df = pd.concat([df, pd.DataFrame({'file_name': score_file, 'trial_type': trial_type,
                                          'frame': np.arange(len(scores)), 'attentional_preference_score': scores})])

    out_file = output_folder.joinpath('preference_scores.csv')
    df.to_csv(out_file, index=False)
    return out_file


def calculate_effect_size(scores_path: Path, output_folder: Path) -> None:
    """
        It calculates the effect size and its statistics for the phonotactics preference test.
        This test calculates the standardised mean gain.
        The coding is such that it is expected to have greater preference for the high probable triphones.
    """
    df = pd.read_csv(scores_path)

    trial_scores = df.groupby(['file_name', 'trial_type']).attentional_preference_score.mean().reset_index(
        name='alpha')

    # calculate the t-statistic, and p-value
    t_statistic, p_value = ttest_ind(trial_scores[trial_scores.trial_type == 'HP'].alpha,
                                     trial_scores[trial_scores.trial_type == 'LP'].alpha, equal_var=False)
    # calculate the effect size
    hp_mean = trial_scores[trial_scores.trial_type == 'HP'].alpha.mean()
    lp_mean = trial_scores[trial_scores.trial_type == 'LP'].alpha.mean()
    hp_std = trial_scores[trial_scores.trial_type == 'HP'].alpha.std()
    lp_std = trial_scores[trial_scores.trial_type == 'LP'].alpha.std()
    hp_n = trial_scores[trial_scores.trial_type == 'HP'].alpha.count()
    lp_n = trial_scores[trial_scores.trial_type == 'LP'].alpha.count()

    d = (hp_mean - lp_mean) / math.sqrt((hp_std ** 2 + lp_std ** 2) / 2)
    g = d * (1 - 3 / (4 * (hp_n + lp_n - 2) - 1))

    # calculate the confidence interval of the effect size
    se_d = (hp_n + lp_n) / (hp_n * lp_n) + (d ** 2) / (2 * (hp_n + lp_n))
    se_g = (hp_n + lp_n) / (hp_n * lp_n) + (g ** 2) / (2 * (hp_n + lp_n))

    # 95% CI
    z = norm.ppf(0.975)  # alpha 0.05
    ci_lb_d = d - z * math.sqrt(se_d)
    ci_ub_d = d + z * math.sqrt(se_d)
    ci_lb_g = g - z * math.sqrt(se_g)
    ci_ub_g = g + z * math.sqrt(se_g)

    df_effects = pd.DataFrame([{'d': d, 't': t_statistic, 'ci_lb_d': ci_lb_d, 'ci_ub_d': ci_ub_d,
                                'se_d': se_d, 'g': g, 'ci_lb_g': ci_lb_g, 'ci_ub_g': ci_ub_g, 'se_g': se_g,
                                'p_value': p_value, 'type': 'phonotactics_preference'}])
    df_effects = df_effects[['type', 'd', 'g', 't', 'se_d', 'se_g', 'ci_lb_d', 'ci_ub_d', 'ci_lb_g', 'ci_ub_g',
                             'p_value']]
    out_file = output_folder.joinpath('output.csv')
    df_effects.to_csv(out_file, index=False)


def calculate_phonotactics_preference_d(preferences_path: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    scores_path = _create_scores_file(preferences_path, output_folder)

    calculate_effect_size(scores_path, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the effect size for the phonotactics preference test.'
                                                 '\nUsage: python calculate_phonotactics_preference_d.py '
                                                 '--input folder_with_preference_numpy_files '
                                                 '--output path_to_output_folder')
    parser.add_argument('--input', type=Path, required=True,
                        help='Path to the folder with the numpy files containing the attentional preference scores per'
                             'frame for each test file')
    parser.add_argument('--output', type=Path, required=True, help='Path to the output folder where to '
                                                                   'save the csv files')
    args = parser.parse_args()

    calculate_phonotactics_preference_d(args.input, args.output)
    sys.exit(0)
