"""
    In order to test vowel discrimination, it is necessary to create two conditions: same vowel and different vowels.
    To do so, this scripts reads corpus info dictionary and create the lists of the two conditions given a list of
    contrasts to be compared.

    @date 25.05.2021
"""

__docformat__ = ['reStructuredText']
__all__ = ['generate_tests_conditions']

import itertools
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Any

IVC_LANGUAGES = ['en', 'nl', 'de', 'fr', 'jp']
IVC_VOWELS = {
    'en': ['A', 'i', 'I', 'E', 'u', '{'],
    'nl': ['I', 'i', 'E', 'u'],
    'de': ['2:', 'I', 'E', 'a:', 'a', '6', 'u:', 'y:'],
    'fr': ['u', 'y', 'a', 'a~', 'i'],
    'jp': ['a', 'u', 'i', 'a:']
}

HC_VOWELS = ['i', 'I', 'E', '{', 'A', 'O', 'U', 'u', 'V', 'Er', 'e', 'o']

OC_VOWELS = ['a', 'a:', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u']


def _check_ivc_parameters(contrasts: List[Tuple[str, str]], contrasts_languages: List[Tuple[str, str]]) -> bool:
    if len(set(language for contrast in contrasts_languages
               for language in contrast).difference(set(IVC_LANGUAGES))) != 0:
        return False

    for idx, contrast in enumerate(contrasts):
        lang1, lang2 = contrasts_languages[idx]
        if contrast[0] not in IVC_VOWELS[lang1]:
            print(contrast)
            return False
        if contrast[1] not in IVC_VOWELS[lang2]:
            print(contrast)
            return False
    return True


def _check_contrasts(contrasts: List[Tuple[str, str]], corpus: str) -> bool:
    corpus_vowels = HC_VOWELS if corpus == 'hc' else OC_VOWELS  # else oc corpus
    for contrast in contrasts:
        if contrast[0] not in corpus_vowels or contrast[1] not in corpus_vowels:
            print(contrast)
            return False
    return True


def _update_list_trials_speakers(filters: List[Tuple[Any, Any]], trials: List[str], speakers: List[str],
                                 trial: str, speaker: str) -> Tuple[List[str], List[str]]:
    for condition, value in filters:
        if condition != value:
            return trials, speakers

    trials.append(trial)
    speakers.append(speaker)
    return trials, speakers


def _apply_filters_trials(corpus_info: dict, filters: dict) -> List[str]:
    trials = []
    for trial in corpus_info.keys():
        log_search = re.search(r'(L\d+)', trial)
        logatome_name = log_search.group(1)

        spkr_search = re.search(r'(S\d+[F,M])', trial)
        speaker_id = spkr_search.group(1)

        var_search = re.search(r'(V\d)', trial)
        variability = var_search.group(1)

        rep_search = re.search(r'(N\d)', trial)
        repetition = rep_search.group(1)

        include_trial = True

        if len(filters['logatomes']) > 0:  # Otherwise include all logatomes
            include_trial = include_trial and (logatome_name in filters['logatomes'])
        if len(filters['speakers']) > 0:
            include_trial = include_trial and (speaker_id in filters['speakers'])
        if len(filters['variability']) > 0:
            include_trial = include_trial and (variability in filters['variability'])
        if len(filters['repetitions']) > 0:
            include_trial = include_trial and (repetition in filters['repetitions'])

        if include_trial:
            trials.append(trial)

    return trials


def _get_trials(corpus_info: dict, contrast: Tuple[str, str], corpus: str, filters: Optional[dict] = None,
                languages: Optional[Tuple[str, str]] = None) -> Tuple[List[str], List[str], List[str], List[str],
                                                                      List[List[str]], List[List[str]]]:
    vowel1, vowel2 = contrast
    trials_v1, trials_v2, speakers_v1, speakers_v2, cvc_v1, cvc_v2 = [], [], [], [], [], []

    if corpus == 'ivc':
        assert languages
        lang1, lang2 = languages
    else:
        lang1, lang2 = None, None

    if corpus == 'hc' and 'failed_listeners_test' in filters:
        include_listeners_test_failed = filters['failed_listeners_test']
    else:
        include_listeners_test_failed = False

    if corpus == 'oc':
        trials = _apply_filters_trials(corpus_info, filters)
    else:
        trials = corpus_info.keys()

    for trial in trials:
        vowel = corpus_info[trial]['vowel']
        speaker = corpus_info[trial]['speaker']
        v1, v2 = False, False
        if vowel == vowel1 or vowel == vowel2:
            if corpus == 'ivc':
                language = corpus_info[trial]['details']['language']
                if vowel == vowel1 and language == lang1:
                    v1 = True
                elif vowel == vowel2 and language == lang2:
                    v2 = True
            elif corpus == 'hc':
                failed_listeners_test = corpus_info[trial]['details']['failed_listeners_test']
                if vowel == vowel1 and include_listeners_test_failed == failed_listeners_test:
                    v1 = True
                elif vowel == vowel2 and include_listeners_test_failed == failed_listeners_test:
                    v2 = True
            else:  # oc
                phones = corpus_info[trial]['details']['phones']
                if vowel == vowel1:
                    v1 = True
                    cvc_v1.append(phones)
                else:
                    v2 = True
                    cvc_v2.append(phones)
            if v1:
                trials_v1.append(trial)
                speakers_v1.append(speaker)
            elif v2:
                trials_v2.append(trial)
                speakers_v2.append(speaker)

    return trials_v1, trials_v2, speakers_v1, speakers_v2, cvc_v1, cvc_v2


def _check_oc_filters(filters: dict) -> dict:
    if not filters:
        filters = {'speakers': [], 'logatomes': [], 'variability': [], 'repetitions': []}

    if 'logatomes' not in filters:
        filters['logatomes'] = []
    if 'speakers' not in filters:
        filters['speakers'] = []
    if 'variability' not in filters:
        filters['variability'] = []
    if 'repetitions' not in filters:
        filters['repetitions'] = []

    return filters


def generate_tests_conditions(corpus_info: dict, contrasts: List[Tuple[str, str]],
                              filters: dict, corpus: str,
                              contrasts_languages: List[Tuple[str, str]]) -> \
        Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
    # check params:
    if corpus == 'ivc':
        assert contrasts_languages
        assert _check_ivc_parameters(contrasts, contrasts_languages)
    elif corpus == 'hc':
        assert _check_contrasts(contrasts, 'hc')
    else:  # oc
        assert _check_contrasts(contrasts, 'oc')
        filters = _check_oc_filters(filters)

    # create two conditions
    same_condition = []
    different_condition = []
    for idx, contrast in enumerate(contrasts):
        if corpus == 'ivc':
            trials_v1, trials_v2, speakers_v1, speakers_v2, _, _ = _get_trials(corpus_info, contrast, 'ivc',
                                                                               languages=contrasts_languages[idx])
        elif corpus == 'hc':
            trials_v1, trials_v2, speakers_v1, speakers_v2, _, _ = _get_trials(corpus_info, contrast, 'hc',
                                                                               filters=filters)
        else:  # oc
            trials_v1, trials_v2, speakers_v1, speakers_v2, cvc_v1, cvc_v2 = _get_trials(corpus_info, contrast,
                                                                                         'oc', filters=filters)

        # same conditions: same vowels
        if corpus != 'oc':
            tmp_same = list(itertools.combinations(trials_v1, 2)) + list(itertools.combinations(trials_v2, 2))
            speakers_tmp_same = list(itertools.combinations(speakers_v1, 2)) + list(itertools.combinations(speakers_v2, 2))

            tmp_different = list(itertools.product(trials_v1, trials_v2))
            speakers_tmp_different = list(itertools.product(speakers_v1, speakers_v2))
        else:
            # Check that trials to be compared have the same cvc context
            trials_v1_cvc = defaultdict(list)
            speakers_v1_cvc = defaultdict(list)
            trials_v2_cvc = defaultdict(list)
            speakers_v2_cvc = defaultdict(list)

            for j, trial in enumerate(trials_v1):
                trials_v1_cvc[''.join(cvc_v1[j])].append(trial)
                speakers_v1_cvc[''.join(cvc_v1[j])].append(speakers_v1[j])
            for j, trial in enumerate(trials_v2):
                trials_v2_cvc[''.join(cvc_v2[j])].append(trial)
                speakers_v2_cvc[''.join(cvc_v2[j])].append(speakers_v2[j])
            tmp_same = []
            speakers_tmp_same = []
            tmp_different = []
            speakers_tmp_different = []
            set_cvc_v2 = set(trials_v2_cvc.keys())
            for cvc in trials_v1_cvc.keys():
                tmp_same += list(itertools.combinations(trials_v1_cvc[cvc], 2))
                speakers_tmp_same += list(itertools.combinations(speakers_v1_cvc[cvc], 2))
                for cvc2 in set_cvc_v2:
                    if cvc[0] == cvc2[0] and cvc[-1] == cvc2[-1]:
                        set_cvc_v2.remove(cvc2)
                        tmp_different += list(itertools.product(trials_v1_cvc[cvc], trials_v2_cvc[cvc2]))
                        speakers_tmp_different += list(itertools.product(speakers_v1_cvc[cvc], speakers_v2_cvc[cvc2]))
                        break
            for cvc in trials_v2_cvc.keys():
                tmp_same += list(itertools.combinations(trials_v2_cvc[cvc], 2))
                speakers_tmp_same += list(itertools.combinations(speakers_v2_cvc[cvc], 2))

        # Across-speaker calculations
        tmp_same = [pair for i, pair in enumerate(tmp_same) if speakers_tmp_same[i][0] != speakers_tmp_same[i][1]]
        same_condition.append(tmp_same)

        tmp_different = [pair for i, pair in enumerate(tmp_different)
                         if speakers_tmp_different[i][0] != speakers_tmp_different[i][1]]
        different_condition.append(tmp_different)

    return same_condition, different_condition
