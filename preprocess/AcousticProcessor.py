# Audio Feature Extractor


from tqdm import tqdm
from utils import confidenceKeys
import pandas as pd
import os
from mmsdk import mmdatasdk
import numpy as np
from sklearn.preprocessing import StandardScaler
import opensmile
import sys

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

words = mmdatasdk.mmdataset({'words': '../data/POM_TimestampedWords.csd'})['words']
#openface = pd.read_csv('./data/POM_TimestampedWords.csd')

# feature names in egemaps
gemaps_n = smile.feature_names
# feature name of flow of speech
fos_n = ['speech_rate_words', 'articulation_words', 'pause_rate', 'pause_ratio', 'pause_mean_dur', 'pause_perc']

# calculate egemaps functional value of each video
def gemaps(file, start=None, end=None):
    y = smile.process_file(file, start=start, end=end)
    return list(y.values[0])


# return speech rate words, articulate rate words, pause rate, pause ratio, pause_mean_dur, pause_perc
def flow_of_speech(f, t):
    pause_duration = 0
    pause_count = 0
    word_count = 0
    len_f = len(f)
    for i in range(0, len_f):
        if f[i][0].decode('UTF-8') == 'sp' and i != 0 and i != len_f-1:
            interval = t[i][1] - t[i][0]
            if interval < 0.5:
                # print('small sp')
                continue
            pause_duration += interval
            pause_count += 1
        if f[i][0].decode('UTF-8') != 'sp':
            word_count += 1
    total_time = t[-1][-1]
    speech_rate_words = (word_count + pause_count) / total_time
    articulation_words = (word_count) / (total_time - pause_duration)
    pause_rate = pause_count / total_time
    pause_ratio = pause_count / word_count
    if pause_count==0:
        pause_mean_dur = 0
    else:
        pause_mean_dur = pause_duration / pause_count
    pause_perc = pause_duration / total_time
    fos =  [speech_rate_words, articulation_words, pause_rate, pause_ratio, pause_mean_dur, pause_perc]
    return fos

# extract acoustic features
def AcousticProcessor():
    keys = confidenceKeys()
    acousticFeature = []
    index = []
    for k in tqdm(keys):
        k = str(k)
        #wav_path = os.path.join('/home/infres/yduan/Stage/data/data_raw/pom/audio_16k', k + '.wav')
        wav_path = os.path.join('../data/audio', k + '.wav')

        f_gemaps = gemaps(wav_path)
        f = np.array(words[k]['features'])
        t = np.array(words[k]['intervals'])
        f_fos = flow_of_speech(f,t)
        acousticFeature.append(f_gemaps+f_fos)
        index.append(k)
    scaler = StandardScaler()
    acousticFeature = scaler.fit_transform(acousticFeature)
    acousticFeature = pd.DataFrame(acousticFeature, columns=gemaps_n+fos_n, index=index)
    acousticFeature.to_csv('../data/acoustic.csv')


if __name__ == "__main__":
    AcousticProcessor()