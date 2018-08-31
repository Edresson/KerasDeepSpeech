#!/usr/bin/env python
#SCRIPT: MOZILLA DEEPSPEECH

'''
    NAME    : LibriSpeech SLR 12
    URL     : http://www.openslr.org/12/
    HOURS   : 1,000
    TYPE    : Read - English
    AUTHORS : Vassil Panayotov et al
    TYPE    : FREE
    LICENCE : CC BY 4.0

'''

from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import codecs
import fnmatch
import pandas
import progressbar
import subprocess
import tarfile
import unicodedata
from pydub import AudioSegment

from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile



def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    '''print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
        TRAIN_CLEAN_360_URL = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
        TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"

        DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
        DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

        TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
        TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

        def filename_of(x): return os.path.split(x)[1]
        train_clean_100 = base.maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)
        bar.update(0)
        train_clean_360 = base.maybe_download(filename_of(TRAIN_CLEAN_360_URL), data_dir, TRAIN_CLEAN_360_URL)
        bar.update(1)
        train_other_500 = base.maybe_download(filename_of(TRAIN_OTHER_500_URL), data_dir, TRAIN_OTHER_500_URL)
        bar.update(2)

        dev_clean = base.maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
        bar.update(3)
        dev_other = base.maybe_download(filename_of(DEV_OTHER_URL), data_dir, DEV_OTHER_URL)
        bar.update(4)

        test_clean = base.maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
        bar.update(5)
        test_other = base.maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)
        bar.update(6)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        LIBRIVOX_DIR = "LibriSpeech"
        work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
        bar.update(0)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
        bar.update(1)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)
        bar.update(2)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
        bar.update(3)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)
        bar.update(4)

        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
        bar.update(5)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)
        bar.update(6)'''

    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    LIBRIVOX_DIR=data_dir 
    with progressbar.ProgressBar(max_value=7,  widget=progressbar.AdaptiveETA) as bar:
        train = _convert_audio_and_split_sentences(LIBRIVOX_DIR)
        bar.update(0)

    # Write sets to disk as CSV files
    train.to_csv(os.path.join(data_dir.replace(LIBRIVOX_DIR.split('/')[-1],''), "train_common_voice.csv"), index=False)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()

def _convert_audio_and_split_sentences(LIBRIVOX_DIR):
        train_label = LIBRIVOX_DIR
	files =[]
	dirf=LIBRIVOX_DIR.split('/')[-2]+'/'+LIBRIVOX_DIR.split('/')[-1]
	#print(train_label,dirf)
        dir_files = LIBRIVOX_DIR.replace(dirf,'')
	#print('dir filess',dir_files)

        for line in codecs.open(train_label, 'r', encoding='utf8'):
	    
                split = line.strip().split(',')
                if split == ['']:
                    continue
                #Ignore first line csv arquive.
                if split[0] == 'filename' :
                    continue

                #print(split[0])
                audio_file = os.path.join(dir_files,split[0])
                transcript = split[1]

                

                transcript = transcript.lower().strip()
		#print	(transcript)
                wav_filesize = os.path.getsize(audio_file)
                sound = AudioSegment.from_file(audio_file, "mp3")
                normalized_sound = match_target_amplitude(sound, -20.0)
                normalized_sound.export(audio_file.replace('.mp3','.wav'), format="wav")
		os.remove(audio_file)
		wav_filesize = os.path.getsize(audio_file.replace('.mp3','.wav'))
                
		
                files.append((audio_file.replace('.mp3','.wav'), wav_filesize, transcript))
	    
		

        return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
