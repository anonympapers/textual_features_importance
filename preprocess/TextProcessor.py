#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs
import spacy
import numpy as np
from ripser import ripser
from persim import plot_diagrams
import PersistenceImages.persistence_images as pimg
from utils import confidenceKeys
from tqdm import tqdm
import pandas as pd







# preprocess raw transcript and split it into sentences
def text2sentence(key):
    with codecs.open('../data/transcript/{}.txt'.format(key), "r", "utf-8") as f:
        lines = f.readlines()
        text = ''
        for l in lines:
            tmp = l.strip()
            if len(tmp) != 0:
                text += tmp + ' '
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    #test_docs = [sent.text.split() for sent in doc.sents]
    return doc


def embeddingExtraction(doc):
    # parameters
    model = "../demo/model.bin"
    # test_docs="toy_data/test_docs.txt"
    #output_file = "./test_vectors.txt"

    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000


    # load Doc2Vec model
    m = g.Doc2Vec.load(model)



    #Build a sample cloud
    sample_cloud = np.array(m.infer_vector(doc.text.split(), alpha=start_alpha, epochs=infer_epoch))

    return sample_cloud




# features extraction
def extractLinkingRate(doc):
    conjunction = dict()
    count_linkers = 0
    for token in doc:
        if (token.pos_ == "CONJ")or(token.pos_ == "CCONJ"):
            count_linkers += 1
            x = conjunction.setdefault(token.lemma_)
            if (x != None):
                conjunction[token.lemma_] += 1
            else:
                conjunction[token.lemma_] = 0
    rate = len(conjunction) / count_linkers
    return rate





def extractSynonymsRate(doc):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet
    from collections import defaultdict
    import pprint
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


    token_dict = dict.fromkeys(doc.text, 1)
    nb_nouns = 0
    syn_cur = []

    for token in doc:
        if (token.pos_ == "NOUN"):
            syn_cur.append(0)
            nb_nouns += 1
            for syn in wordnet.synsets(token.text):
                for i in syn.lemmas():
                    if token_dict.setdefault(i.name()) != None:
                        syn_cur[-1] += 1


    syn_cur = [elem / (nb_nouns*nb_nouns) for elem in syn_cur]
    return sum(syn_cur)



# def extractSUPERT():
#     from ref_free_metrics.supert import Supert
#     from utils.data_reader import CorpusReader
#
#     print("Hi")
#
#     # read docs and summaries
#     reader = CorpusReader('data/topic_1')
#     source_docs = reader()
#     summaries = reader.readSummaries()
#
#     # compute the Supert scores
#     supert = Supert(source_docs, ref_metric='top15')
#     scores = supert(summaries)
#
#     print(scores)



# giving extracted features to the TextProcess()
def extractTextFeatures(doc):
    linking_rate = extractLinkingRate(doc)
    synonyms_rate = extractSynonymsRate(doc)
    embedding = embeddingExtraction(doc)

    features_rate = [linking_rate, synonyms_rate]
    features_embedding = embedding


    return features_rate, features_embedding




def TextProcess():
    # iterate by the text files
    keys = confidenceKeys()
    featureRateName = ['linking_rate', 'synonyms_rate']
    featureEmbedName = ['emb_axis_' + str(i) for i in range(100)]

    # featureName.extend(['emb_axis_' + str(i) for i in range(100)])
    textRateFeature = []
    textEmbedFeature = []

    index = []
    for k in tqdm(keys):
        doc = text2sentence(k)
        rateFeature, embedFeature = extractTextFeatures(doc)
        #extractSUPERT()
        textRateFeature.append(rateFeature)
        textEmbedFeature.append(embedFeature)

        index.append(k)

    # scalar = StandardScaler()
    # textFeature = scalar.fit_transform(textFeature)
    textRateFeature = pd.DataFrame(textRateFeature, columns=featureRateName, index=index)
    textEmbedFeature = pd.DataFrame(textEmbedFeature, columns=featureEmbedName, index=index)

    textRateFeature.to_csv('../data/text_rate.csv')
    textEmbedFeature.to_csv('../data/text_embeddings.csv')

    




if __name__ == "__main__":
    TextProcess()