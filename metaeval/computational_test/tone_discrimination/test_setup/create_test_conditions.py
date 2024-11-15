"""
    @author María Andrea Cruz Blandón
    @date 02.11.2023

    This script create the list of pair of files that should be used for the DTW distance calculation for the same
    and different condition. This depends on the test type.
"""

import itertools
from collections import defaultdict
from typing import List, Tuple

__docformat__ = ['reStructuredText']
__all__ = ['generate_test_conditions']


def generate_test_conditions(index_info: dict, test_type: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
        It creates the list with pairs of files to calculate DTW distances. It calculates all the possible combinations
        such they are for the same syllable and different speaker for the same tone or different tone.
    """

    same_condition = []
    different_condition = []

    # Get all the info of the files
    files_info = {}

    for key in index_info.keys():
        if test_type == 'basic':
            if key.split('_')[0][:-1] != 'ci':
                continue
        files_info[key] = {'syllable': key.split('_')[0][:-1],
                           'speaker': key.split('_')[1],
                           'tone': key.split('_')[0][-1]}

    trials_tone25 = defaultdict(list)
    trials_tone33 = defaultdict(list)

    for key, value in files_info.items():
        if value['tone'] == '2':
            trials_tone25[value['syllable']].append(key)
        else:
            trials_tone33[value['syllable']].append(key)

    for syllable in trials_tone25.keys():
        same_condition += list(itertools.combinations(trials_tone25[syllable], 2))
        same_condition += list(itertools.combinations(trials_tone33[syllable], 2))
        different_condition += list(itertools.product(trials_tone25[syllable], trials_tone33[syllable]))

    # remove those pairs that are from the same speaker
    same_condition = [pair for pair in same_condition if
                      files_info[pair[0]]['speaker'] != files_info[pair[1]]['speaker']]

    different_condition = [pair for pair in different_condition if
                           files_info[pair[0]]['speaker'] != files_info[pair[1]]['speaker']]

    return same_condition, different_condition
