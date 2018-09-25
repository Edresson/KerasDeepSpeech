from pydub import AudioSegment
import os

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
data = 'LibriSpeech/train-clean-100-wav/'
files=os.listdir(data)
for file in files:
    sound = AudioSegment.from_file(data+file, "wav")
    normalized_sound = match_target_amplitude(sound, -20.0)
    normalized_sound.export(data+file, format="wav")
