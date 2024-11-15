"""
    @author María Andrea Cruz Blandón
    @date 13.09.2023

    Given a directory with the numpy files containing the attentional preference scores per frame for each test file,
    it calculates the IDS preference effect size (d).

    The folder structure should be
    Test
    - IDS
    - - file1.npy
    - - file2.npy
    - - ...
    - ADS
    - - file1.npy
    - - file2.npy
    - - ...

    It will produce an intermediate csv file with the information of all files in the test, with the following format:
    file_name, trial_type, frame, attentional_preference_score
    ../file1, IDS, 0, 0.40
    ../file1, IDS, 1, 0.42
    ../file2, ADS, 0, 0.32
    ../file2, ADS, 1, 0.35
    ../file2, ADS, 2, 0.32


    Finally, the output file will have the following format:
    type, d, g, t, se_d, se_g, ci_lb_d, ci_ub_d, ci_lb_g, ci_ub_g, p_value

    where type is ids_preference
"""

import argparse
import math
import sys

import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind, norm

__docformat__ = ['reStructuredText']
__all__ = ['calculate_ids_preference_d', 'calculate_statistics', 'write_final_csv']


def calculate_statistics(scores: pd.DataFrame) -> dict:
    """
        It calculates the t-statistic, p-value, confidence interval of the effect size and the effect size
        (standardised mean gain) for the IDS preference test.
    """
    # get the scores per trial
    trial_scores = scores.groupby(['file_name', 'trial_type']).attentional_preference_score.mean().reset_index(
        name='alpha')

    # calculate the t-statistic, and p-value
    t_statistic, p_value = ttest_ind(trial_scores[trial_scores.trial_type == 'IDS'].alpha,
                                     trial_scores[trial_scores.trial_type == 'ADS'].alpha, equal_var=False)
    # calculate the effect size
    ids_mean = trial_scores[trial_scores.trial_type == 'IDS'].alpha.mean()
    ads_mean = trial_scores[trial_scores.trial_type == 'ADS'].alpha.mean()
    ids_std = trial_scores[trial_scores.trial_type == 'IDS'].alpha.std()
    ads_std = trial_scores[trial_scores.trial_type == 'ADS'].alpha.std()
    ids_n = trial_scores[trial_scores.trial_type == 'IDS'].alpha.count()
    ads_n = trial_scores[trial_scores.trial_type == 'ADS'].alpha.count()

    d = (ids_mean - ads_mean) / math.sqrt((ids_std ** 2 + ads_std ** 2) / 2)
    g = d * (1 - 3 / (4 * (ids_n + ads_n - 2) - 1))

    # calculate the confidence interval of the effect size
    se_d = (ids_n + ads_n) / (ids_n * ads_n) + (d ** 2) / (2 * (ids_n + ads_n))
    se_g = (ids_n + ads_n) / (ids_n * ads_n) + (g ** 2) / (2 * (ids_n + ads_n))

    # 95% CI
    z = norm.ppf(0.975)  # alpha 0.05
    ci_lb_d = d - z * math.sqrt(se_d)
    ci_ub_d = d + z * math.sqrt(se_d)
    ci_lb_g = g - z * math.sqrt(se_g)
    ci_ub_g = g + z * math.sqrt(se_g)

    return {'d': d, 't': t_statistic, 'ci_lb_d': ci_lb_d, 'ci_ub_d': ci_ub_d,
            'se_d': se_d, 'g': g, 'ci_lb_g': ci_lb_g, 'ci_ub_g': ci_ub_g, 'se_g': se_g, 'p_value': p_value,
            'type': 'ids_preference'}


def _create_scores_file(attentional_scores: Path, output_path: Path) -> Path:
    """
        It creates an intermediate csv file with the information of all files in the test with the attentional
        preference scores per frame.
    """
    assert attentional_scores.exists(), f'Attentional scores directory {attentional_scores} does not exist'
    assert attentional_scores.joinpath(
        'IDS').exists(), f'IDS directory {attentional_scores.joinpath("IDS")} does not exist'
    assert attentional_scores.joinpath(
        'ADS').exists(), f'ADS directory {attentional_scores.joinpath("ADS")} does not exist'

    files = attentional_scores.rglob('**/*.npy')

    df = pd.DataFrame(columns=['file_name', 'trial_type', 'frame', 'attentional_preference_score'])

    for score_file in files:
        trial_type = score_file.parent.stem
        scores = np.load(score_file).reshape(-1)
        df = pd.concat([df, pd.DataFrame({'file_name': score_file, 'trial_type': trial_type,
                                         'frame': np.arange(0, len(scores)), 'attentional_preference_score': scores})])

    output_path = output_path.joinpath('preference_scores.csv')

    df.to_csv(output_path, index=False)

    return output_path


def write_final_csv(metrics: dict, output_folder: Path) -> None:
    out_df = pd.DataFrame.from_records([metrics], columns=['type', 'd', 'g', 't', 'se_d', 'se_g', 'ci_lb_d', 'ci_ub_d',
                                                           'ci_lb_g', 'ci_ub_g', 'p_value'])
    output_file = output_folder.joinpath('output.csv')
    out_df.to_csv(output_file, index=False)

    print(f'Output file {output_file} created')


def calculate_ids_preference_d(attentional_scores: Path, output_folder: Path) -> None:
    """
        Calculates the effect size, the standardised mean gain, and it saves the results intermediate and final in
        csv format.
    """
    # Create intermediate csv file
    output_folder.mkdir(parents=True, exist_ok=True)
    intermediate_result = _create_scores_file(attentional_scores, output_folder)
    # Calculate effect size
    df = pd.read_csv(intermediate_result)
    metrics = calculate_statistics(df)
    write_final_csv(metrics, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the IDS preference effect size (d) ,and outputs the '
                                                 'result in a csv file.'
                                                 '\nUsage: python calculate_ids_preference_d.py '
                                                 '--input <directory with numpy files> '
                                                 '--output <output_folder_csv_file>')
    parser.add_argument('--input', required=True,
                        type=Path, help='The path to directory with numpy files.')
    parser.add_argument('--output', required=True,
                        type=Path, help='The path to the directory where to save the output csv files')

    args = parser.parse_args()

    calculate_ids_preference_d(args.input, args.output)
    sys.exit(0)
