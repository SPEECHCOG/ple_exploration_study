"""
    This script generates the tokens for the discrimination test of tone 25 and tone 33 of Cantonese.
"""

from google.cloud import texttospeech
from pathlib import Path

client = texttospeech.TextToSpeechClient()

output_path = Path('tone_data')
output_path.mkdir(parents=True, exist_ok=True)

voices = client.list_voices().voices
cantonese_voices = []
for voice in voices:
    if 'yue-HK' in voice.language_codes:
        cantonese_voices.append(voice)

audio_config = texttospeech.AudioConfig({'audio_encoding': texttospeech.AudioEncoding.MULAW})

with open(Path('./cantonese_minimal_pairs.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        tone_25, tone_33 = line.strip().split('\t')

        synthesis_input_25 = texttospeech.SynthesisInput({
            'ssml': f"<speak><phoneme alphabet=\"jyutping\" ph=\"{tone_25}\"></phoneme></speak>"})
        synthesis_input_33 = texttospeech.SynthesisInput({
            'ssml': f"<speak><phoneme alphabet=\"jyutping\" ph=\"{tone_33}\"></phoneme></speak>"})

        for voice in cantonese_voices:
            syn_voice = texttospeech.VoiceSelectionParams({
                'language_code': 'yue-HK',
                'name': voice.name,
                'ssml_gender': voice.ssml_gender
            })

            response_25 = client.synthesize_speech(input=synthesis_input_25, voice=syn_voice, audio_config=audio_config)
            response_33 = client.synthesize_speech(input=synthesis_input_33, voice=syn_voice, audio_config=audio_config)
            with open(output_path.joinpath(f'{tone_25}_{voice.name}.wav'), 'wb') as out:
                out.write(response_25.audio_content)
            with open(output_path.joinpath(f'{tone_33}_{voice.name}.wav'), 'wb') as out:
                out.write(response_33.audio_content)


