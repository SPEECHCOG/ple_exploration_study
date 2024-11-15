"""
    @author María Andrea Cruz Blandón
    @date 13.09.2023

    This script calls get_attentional_scores to calculate the scores for a set of models and creates the output per
    each model copying the directory structure of the directory of models.
"""
import argparse
import sys
from pathlib import Path

__docformat__ = ['reStructuredText']
__all__ = ['obtain_all_attentional_scores']

from get_attentional_scores import get_attentional_scores


def obtain_all_attentional_scores(input_feats: str, models_dir: str, config: str, overlap: float,
                                  output_path: str, lang: str = None) -> None:
    """
        Get all models by filtering using .h5 extension and call get_attentional_scores to calculate the scores
        for each model.
    """
    # Get all models
    all_models = Path(models_dir).rglob('**/*.h5')
    lang_options = {'en': 'english', 'fr': 'french'}

    for model in all_models:
        if lang is not None:
            if lang_options[lang] not in str(model):
                continue
        output_file = Path(output_path).joinpath(model.relative_to(models_dir)).with_suffix('.csv')
        get_attentional_scores(input_feats, model, config, overlap, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the attentional scores for a set of models.\nUsage: '
                                                 'python obtain_all_attentional_scores.py '
                                                 '--input path_to_input_features '
                                                 '--models path_dir_models '
                                                 '--config model_configuration '
                                                 '--overlap float_overlap_frames'
                                                 '--output dir_output '
                                                 '--lang <en|fr>')
    parser.add_argument('--input', required=True, type=str, help='Path to the input features')
    parser.add_argument('--models', required=True, type=str, help='Path to the directory with the models')
    parser.add_argument('--config', required=True, type=str, help='Path to the YAML model configuration file')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap of frames for loss calculation')
    parser.add_argument('--output', required=True, type=str, help='Path to the output directory')
    parser.add_argument('--lang', type=str, help='Language of the models', choices=['en', 'fr'])
    args = parser.parse_args()
    obtain_all_attentional_scores(args.input, args.models, args.config, args.overlap, args.output, args.lang)
    sys.exit(0)
