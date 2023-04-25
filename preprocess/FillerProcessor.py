# Extract Fluency Related Features

import pandas as pd
from utils import confidenceKeys
import spacy
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# preprocess raw transcript and split it into sentences
def text2sentence(key):
    with open('../data/transcript/{}.txt'.format(key)) as f:
        lines = f.readlines()
        text = ''
        for l in lines:
            tmp = l.strip()
            if len(tmp) != 0:
                text += tmp + ' '
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    return sents


# extract filler features of each transcript
def extractFillerFeatures(sents):
    f_start = 0
    f_mid = 0
    f_uncertain = 0
    f_sen = 0
    cnt_sen = 0
    f_uh = 0
    f_um = 0
    f_tot = 0
    for s in sents:
        tokens = s.split()
        tmp_len = len(tokens)
        tmp_start = s[:5].count('hh)')
        tmp_uh = s.count('(uhh)') + s.count('ahh')
        tmp_um = s.count('(umm)')
        tmp_mid = tmp_uh + tmp_um - tmp_start
        f_start += tmp_start
        f_mid += tmp_mid
        if s.count('stutter'):
            f_uncertain += tmp_uh + tmp_um
        if tmp_mid:
            cnt_sen += 1
            f_sen += tmp_len
        f_uh += tmp_uh
        f_um += tmp_um
        f_tot += tmp_len
    f_start /= f_tot
    f_mid /= f_tot
    f_uncertain /= f_tot
    f_sen /= cnt_sen+1e-12
    f_uh /= f_tot
    f_um /= f_tot
    return [f_uh, f_um, f_start, f_mid, f_uncertain, f_sen]


# extract filler related features
def FillerProcess():
    keys = confidenceKeys()
    featureName = ['f_uh', 'f_um', 'f_start', 'f_mid', 'f_uncertain', 'sen_len']
    fillerFeature = []
    index = []
    for k in tqdm(keys):
        sents = text2sentence(k)
        features = extractFillerFeatures(sents)
        fillerFeature.append(features)
        index.append(k)
    scalar = StandardScaler()
    fillerFeature = scalar.fit_transform(fillerFeature)
    fillerFeature = pd.DataFrame(fillerFeature, columns=featureName, index=index)
    fillerFeature.to_csv('../data/filler.csv')


if __name__ == "__main__":
    FillerProcess()