"""
    @author María Andrea Cruz Blandón
    @date 06.11.2023

    This scripts uses the Google Cloud API to create the corpora for the phonotactics test. The triphones used are those
    used in Gonzalez-Gomez et al., 2021 and originally in Jusczyk et al., 1994. Totalling 12 triphones for higher
    probability (HP) and 12 for lower probability (LP).
"""

from google.cloud import texttospeech
from pathlib import Path

client = texttospeech.TextToSpeechClient()

output_path = Path('phonotatics_data')
output_path.mkdir(parents=True, exist_ok=True)

output_path.joinpath('HP').mkdir(parents=True, exist_ok=True)
output_path.joinpath('LP').mkdir(parents=True, exist_ok=True)

voices = client.list_voices().voices
american_english_voices = []
for voice in voices:
    if 'en-US' in voice.language_codes:
        american_english_voices.append(voice)

audio_config = texttospeech.AudioConfig({'audio_encoding': texttospeech.AudioEncoding.MULAW})

# IPA transcriptions
hp_triphones = ['dʌs', 'kʌn', 'pʌm', 'sʌl', 'bis', 'dis', 'pim', 'ɹin', 'keb', 'pek', 'sed', 'tes']
lp_triphones = ['ʧʌʃ', 'ʧeg', 'wʌʧ', 'jʌdʒ', 'jiʃ', 'øʌv', 'øeð', 'giʃ', 'ziʧ', 'ziø', 'ʃeg', 'ʃeø']

for idx in range(len(hp_triphones)):
    synthesis_hp = texttospeech.SynthesisInput({
        'ssml': f"<speak><phoneme alphabet=\"ipa\" ph=\"{hp_triphones[idx]}\"></phoneme></speak>"})
    synthesis_lp = texttospeech.SynthesisInput({
        'ssml': f"<speak><phoneme alphabet=\"ipa\" ph=\"{lp_triphones[idx]}\"></phoneme></speak>"})

    for voice in american_english_voices:
        syn_voice = texttospeech.VoiceSelectionParams({
            'language_code': 'en-US',
            'name': voice.name,
            'ssml_gender': voice.ssml_gender
        })

        response_hp = client.synthesize_speech(input=synthesis_hp, voice=syn_voice, audio_config=audio_config)
        response_lp = client.synthesize_speech(input=synthesis_lp, voice=syn_voice, audio_config=audio_config)

        with open(output_path.joinpath('HP').joinpath(f'hp-{idx}_{voice.name}.wav'), 'wb') as out:
            out.write(response_hp.audio_content)

        with open(output_path.joinpath('LP').joinpath(f'lp-{idx}_{voice.name}.wav'), 'wb') as out:
            out.write(response_lp.audio_content)

