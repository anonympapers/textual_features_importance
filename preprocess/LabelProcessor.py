# Select high and low confidence videos from original dataset

from mmsdk import mmdatasdk

import pandas as pd
import pickle
from utils import confidenceKeys

# extract high and low confidence label data
def extractConfidenceLabels():
    #cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, 'cmumosi/')
    #POM_data = mmdatasdk.mmdataset(mmdatasdk.pom.highlevel['OpenFace'], 'data/')


    #infile = open('./data/HPOM_train.pkl', 'rb')
    #data = pd.read_pickle(infile)

    #print(type(data))


    labels = mmdatasdk.mmdataset({'labels': '../data/POM/POM_Labels.csd'})['labels']
    #openface = mmdatasdk.mmdataset({'openface': './data/POM_OpenFace2.csd'})['openface']
    #all_keys = list(openface.keys())
    openface = pd.read_csv('../data/openface_ID.csv')
    all_keys = openface['0'].values.tolist()
    data = pd.DataFrame()
    pers_keys = confidenceKeys()

    # print(pers_keys)

    for key in labels.keys():
        # print(key)
        if int(key) in pers_keys:
            # if key not in all_keys:
            #     # print('rej')
            #     continue
            labs = labels[key]['features'][0]

            # print(labs[0], labs[1], labs[2])

            # extract confident and persuasion [confidence_score, persuasiveness_score]
            if labs[0] >= 4:
                data = data.append(pd.DataFrame([1], columns=['label'], index=[key]))
            if labs[0] < 4:
                data = data.append(pd.DataFrame([0], columns=['label'], index=[key]))

    print(data['label'].value_counts())
    # print(len(pers_keys))
    data.to_csv('../data/confidenceLabel.csv')

if __name__ == "__main__":
    extractConfidenceLabels()