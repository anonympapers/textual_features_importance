# run demo mock-up. Input Id of the video and generate related feedback JSON file
#
# from flair.data import Sentence
# from flair.models import SequenceTagger
import argparse
import gensim.models as g
import codecs
import spacy
import numpy as np
from utils import confidenceKeys
from tqdm import tqdm
import pandas as pd
import sys
import os
import math
global dataset
global language

sys.path.append('/Users/goddessoffailures/Desktop/Mock-Up_sample/Mock-Up_sample/REVITALISE')

#---------------------------------------------------------------------
# Word Level Features of Lexical diversity
#---------------------------------------------------------------------
# Those features are taken from article
# paper: Automated assessment of non-native learner essays: Investigating the role of linguistic features
# authors: Sowmya Vajjala, Iowa State University, USA sowmya@iastate.edu
def nbTokens(doc):
    return len(doc)

def nbType(doc):
    types = tokenTypes(doc)
    return len(types)

def tokenTypes(doc):
    types = dict()
    for token in doc:
        x = types.setdefault(token.pos_)
        if x == None:
            types[token.pos_] = 1
        else:
            types[token.pos_] += 1
    return types


#---------------------------------------------------------------------
# Preprocess of creation of full french tagging
#---------------------------------------------------------------------

# This function takes doc = nlp(transcript) with fr_core_news_sm
# And gives a list of lists where each one is (word, lemma, POS, general tag)
#TODO: lowercase !!!! for lemmas at least!!

class frencToken():
    def __init__(self, text = None, lemma = None, pos = None, tag = None):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag

class frenchDoc():
    def __init__(self, listOfFrenchSentences):
        self.docFull = []
        self.text = ''
        for sent in listOfFrenchSentences:
            self.docFull.extend(sent)
            for word in sent:
                self.text += word.text + ' '

        self.docPerSent = listOfFrenchSentences


def fullTaggingForFrench(transcript):

    # Load the model
    # model = SequenceTagger.load("qanastek/pos-french")
    nlp = spacy.load('fr_core_news_sm')

    # List of sentences
    taggedDocBySent = []

    #TODO: because of automatic annotation there is no sentences
    # Time: 180 sec
    # average length of sentence in french (words): 10-15
    transcript = transcript.split(" ")
    nbOfWords = len(transcript)
    end = round(nbOfWords/10)
    sentences = []

    for i in range(0, end):
        if (i != end ):
            sentByWord = transcript[i*10: (i+ 1)*10]
        else:
            sentByWord = transcript[i:]

        sentences.append(" ".join(sentByWord))

    # sentences = transcript.split('.')

    listOfSentences = []

    for sentence in sentences:
        # # Create sentence object out of doc
        # string = Sentence(sentence)
        # # Predict tags
        # # prediction = model.predict(string)
        # # Split tagged object to get list of pairs word-tag
        # taggedText = string.to_tagged_string().split("→")[1].split(", ")
        # # Creates a list of pairs (token_text, token_tag)
        # taggedText = [words.replace(' ', '').replace('\"', '').replace('\\', '').replace('[', '').replace(']', '').split("/")[1] for words in taggedText]
        # # extract POS via spacy
        spacyPOS = nlp(sentence)
        # Create a list of tokens in the sentence
        listOfTokens = [ frencToken(spacyToken.text, spacyToken.lemma_, spacyToken.pos_, None) for spacyToken in spacyPOS]
        listOfSentences.append(listOfTokens)

    doc = frenchDoc(listOfSentences)
    return doc








#---------------------------------------------------------------------
# Preprocess raw transcript and split it into sentences
#---------------------------------------------------------------------

#TODO: takenizer nlp has to tale eng and fr options now on;y french works
def text2sentence(key):
    with codecs.open('../data/'+ dataset +'/transcripts/' + key, "r", "utf-8") as f:
        lines = f.readlines()
        text = ''
        for l in lines:
            tmp = l.strip()
            if len(tmp) != 0:
                text += tmp + ' '
    nlp = spacy.load('fr_core_news_sm')
    #nlp.pipe_labels['tagger']

    if (language == 'fr'):
        doc = fullTaggingForFrench(text)
    else: 
        doc = nlp(text)

    # print([token.tag_ for token in doc ])
    # test_docs = [sent.text.split() for sent in doc.sents]
    return doc

#---------------------------------------------------------------------
# Extract pos/neg polarity features and provide to the extractTextFeatures()
#---------------------------------------------------------------------



def polarDictionary(file):
    # nlp = spacy.load('fr_core_news_sm')
    data = np.array(pd.read_csv(file, header = 0).values)
    X_dataArr = ''
    Y_dataArr = []
    for i in data:
        X_dataArr = X_dataArr + ' ' + str(i[0])
        Y_dataArr.append(i[1])
    docposneg = fullTaggingForFrench(X_dataArr)
    dataDict = {key.text: val for key, val in zip(docposneg.docFull, Y_dataArr )}
    # dataDict = dict.fromkeys(X_dataArr, Y_dataArr)
    return dataDict





def posNegPolarity(doc, polarDict):
    # print("posNegPolarity   start")

    listOfFeatures = ['posTerms', 'negTerms', 'neutralTerms']
    polarityFeatures = dict.fromkeys(listOfFeatures, 0)

    for token in doc.docFull:
        # print(token.text)
        if (polarDict.setdefault(token.text) == 1):
            polarityFeatures['posTerms'] += 1
        elif (polarDict.setdefault(token.text) == 0):
            polarityFeatures['negTerms'] += 1
        else:
            polarityFeatures['neutralTerms'] += 1

    # print(polarityFeatures)
    return polarityFeatures


#---------------------------------------------------------------------
# Extract all features and provide to the TextProcess()
#---------------------------------------------------------------------

def extractTextFeatures(doc, polarDict):
    if (language == 'fr'):
        polarity = posNegPolarity(doc, polarDict)
    else:
        print('English version of tagging etraction is not concistent with french one')

    return polarity



#---------------------------------------------------------------------
# main function to extract and save results
#---------------------------------------------------------------------

def TextProcess():
    dir_path = '../data/' + dataset +'/transcripts/'

    polarDict = polarDictionary('../data/MT/lexicon/pos_neg_words_fr.csv')
    keys = os.listdir(dir_path)
    confidenceKeys()
    featurePolarity = ['posTerms', 'negTerms', 'neutralTerms']
    textPolarity = []
    index = []
    for k in tqdm(keys):
        # print('start')
        doc = text2sentence(k)
        ratePolarity = extractTextFeatures(doc, polarDict)
        textPolarity.append(ratePolarity)
        index.append(os.path.splitext(k)[0])

    textPolarity = pd.DataFrame(textPolarity, columns=featurePolarity, index=index)
    textPolarity.to_csv('../data/' + dataset +'/textPolarity.csv')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This program extracts textual features and requires the specification of dataset: MT or POM')
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    if (dataset == 'MT'):
        language = 'fr'
    else:
        language = 'eng'


    TextProcess()


