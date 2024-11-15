# Simulating Prenatal Language Exposure 

This is the repository to replicate the results from the paper "Simulating Prenatal 
Language Exposure in Computational Models: An Exploration Study".

For replication from the precomputed results, you can directly follow the instructions
in [Replicating plots](#replicating-plots). Otherwise, you will find in this repository 
the code used to [filter the audio files](#filtering-audio-files), [extract the acoustic
features](#extracting-acoustic-features), [train the models](#training-models), and
[run the linguistic test](#running-linguistic-tests).

## Requirements
The code was developed using python 3.9, tensorflow 2.4, Matlab R2019b, and R version 4.2.0

To execute the python code please create a new conda environment from the `requirements.yml`
file. For that run:

```bash
conda env create -f requirements.yml
```

Then you can activate the environment by executing:

```bash
conda activate ple_env
```

## Replicating Plots
To create figures 4 and 5 in the manuscript first go to `replication_plots` folder and then
open in R Studio the script `plot_results.R`.

The replication folder contains the infant reference data (`reference_data`) for each of the four tests: IDS 
preference, Phonotactics preference, Tone discrimination and Vowel discrimination. Within the same
folder, there is the reference tables use to create figures 6 and 7 in the manuscript in 
`comparison_tables`.

The folder `results` contains all the effect sizes obtain by the APC and CPC models in each of the
linguistic tests for the 3 runs reported in the paper. 

## Filtering Audio Files
We simulated the womb filtering effect with a filter in Matlab. We based our filter in 
(Gerhardt and Abrams, 1996). The matlab code to filter the signals is located in the folder
`filtering`. 

You can apply the filter to a folder with `wav` files as follows:

```bash
cd filtering
matlab -nodisplay -r "filter_files(<path_to_source_dir>, <path_to_target_dir>); exit"
```

The filter files will be in `wav` format and the directory structure will be preserved.

## Extracting Acoustic Features
For our experiments, we extracted 40 logmel features from the audio files (original and 
filtered signals), with a window length of 25 ms and a window shift of 10 ms. An example 
file of the configuration is found in `feature_extraction/preprocess_input_data.json`. 

To extract the features you will need to execute the python script `preprocess_training_data.py`

```bash
cd feature_extraction
python preprocess_training_data.py --config <path_to_json_file>
```

This will create `h5` files with the features in the folder established in the configuration
file. 

Since the models are trained in an incremental way where each uterance is only shown once during
the training, an index file is generated for that process. To generate this index file 
use the script `create_dataset_summary.py`

```bash
python create_dataset_summary.py --input <path_to_folder_with_features> --output <path_to_csv_summary_file>
```

Then the train and validation dataset can be calculated using `split_data.py`. Adjust the script
with your data, In our case, we used 2.5 hours of speech for validation. 

## Training Models
We implemented autoregressive predictive coding (APC) (Chung et al., 2019) and contrasting predictive coding (CPC)
(van den Oord et al., 2018). The training was done in an incremental fashion in which each utterance was used
only once during the training. 

All the code related to training and extracting internal representations and attentional scores
from the models is located in the folder `predictive models`.

For training the models set the configuration using the models `config_apc_en.yml` and `config_cpc_en.yml`
specifying the training data both original and filtered, the profile: `sanity_check_original` (regular training only
use full signal for training), `ale` (continue training from a model that has been trained with filtered signal), or 
`ple` (train model with filtered signal), the checkpoints you are interested in saving, and the model parameters. 

```bash 
cd predictive_models
python train.py --config <path_to_configuration_file>
```

Use `predict.py` to generate the internal representations (or get latent representations), and 
`get_attentional_scores.py` to generate the attentional scores (or get loss values per frame). This is used for the 
linguistic tests as explained in the following section. 

```bash
python predict.py --config <path_to_configuration_file> --model <path_to_model_checkpoint> --input <path_to_features_files> --output <path_to_npy_output_file>
python get_attentional_scores.py --config <path_to_configuration_file> --model <path_to_model_checkpoint> --input <path_to_features_files> --output <path_to_npy_output_file>
```
The configuration file is the same used for training the model. 

## Running Linguistic Tests
We run four linguistic tests: Infant-directed speech (IDS) preference, phonotactics preference, tone 
discrimination, and native vowel discrimination. We used the tests in [MetaEval](https://github.com/SPEECHCOG/metaeval)
(Cruz Blandón et al., 2023) and copy the used version to this repository in `metaeval` folder. 

All the test data is located in each test folder under the folder `test_data`, please extract the acoustic features for 
each of them according to those used during training. Once you have the features for each test you can calculate model 
effect sizes using the results of `predict.py` and `get_attentional_scores.py` as needed. 

Activate the conda environment of metalab

```bash
cd metaeval
conda env create -f requirements.yml
conda activate metaeval
```

### Preference tests
For this type of test, you need to calculate the attentional scores. Once you have the numpy files (`npy`) for the test
data, you can execute the tests:

#### IDS preference
```bash 
cd metaeval/computational_test/ids_preference
python calcualte_ids_preference_d.py --input <path_to_npy_files_attentional_scores> --output <path_to_output_folder>
```

#### Phonotactics preference

```bash
cd metaeval/computational_test/phonotactics_preference
python calculate_phonotactics_preference_d.py --input <path_to_npy_files_attentional_scores> --output <path_to_output_folder>
```

This will create an intermediate csv file with the attentional scores per file and frame. Then it will create a csv files
with the information about the effect size calculated (d, g, and t metrics).

### Discrimination tests

For this type of test, you need to calculate the internal representations, use `predict.py` script. Once you have the 
numpy files (`npy`) with the internal representations for the test data, you can execute the tests:

#### Native vowel discrimination

```bash
cd metaeval/computational_test/vowel_discrimination
python calculate_vowel_discrimination_d.py --predictions <path_to_predictions> --type native --output <path_to_output_folder>
```

#### Tone discrimination
```bash
cd metaeval/computational_test/tone_discrimination
python calculate_tone_discrimination_d.py --predictions <path_to_predictions> --type extended --output <path_to_output_folder>
```

This will create an intermediate csv files with the distances between two pairs (tones or vowels) and then it will create
a csv file with the effect size (d, g, and t metrics)

## References

Chung, Y. A., Hsu, W. N., Tang, H., & Glass, J. (2019). An unsupervised autoregressive model for speech representation learning. Proceedings of the Annual Conference of the International Speech Communication Association (Interspeech), pp. 146–150. https://doi.org/10.21437/Interspeech.2019-1473

Cruz Blandón, M. A., Cristia, A., & Räsänen, O. (2023). Introducing meta-analysis in the evaluation of computational models of infant language development. Cognitive Science, 47 (7), e13307. https://doi.org/https://doi.org/10.1111/cogs.13307

Gerhardt, K. J., & Abrams, R. M. (1996). Fetal hearing: Characterization of the stimulus and response [Vibration Exposure in Pregnancy]. Seminars in Perinatology, 20 (1), 11–20. https://doi.org/https://doi.org/10.1016/S0146-0005(96)80053-X

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding. Computing Research Repository, abs/1807.0. https://arxiv.org/abs/1807.03748

The ManyBabies Consortium. (2020). Quantifying sources of variability in infancy research using the infant-directed speech preference. Advances in Methods and Practices in Psychological Science (AMPPS), 3(1), 24-52. DOI: 10.1177/2515245919900809

James Hillenbrand, Laura A. Getty, Michael J. Clark, and Kimberlee Wheeler , "Acoustic characteristics of American English vowels", The Journal of the Acoustical Society of America 97, 3099-3111 (1995) https://doi.org/10.1121/1.411872

## Citing this work
María Andrea Cruz Blandón, Nayeli Gonzalez-Gomez, Marvin Lavechin, and Okko Räsänen. (2024). Simulating Prenatal Language Exposure in Computational Models: An Exploration Study. Submitted

## Contact
If you find any issue please report it on the issues section in this repository. Further comments can be sent to maria <dot> cruzblandon <at> tuni <dot> fi