
# SPSTS

Thank you for visiting our "Shapley Value Based Public Speaking Training System" repository. Here you can find features extractors, models, and results. This repository is anonymous thus we do not provide initial data we were working with. In the case of any questions and usage of our code we kindly ask you to contact us via e-mail address: anonym.papers@gmail.com




## Section Content

This repository contains following:

* __requirements.txt__

* __feature_category.py__: produces file __categories.json__ to be used for separation features on groups

* __test_MT.py__ file: 
    * group features from different categories using __categories.json__
    * learn & evaluate classifier
    * conduct shap analysis
    
* __./data/__:
    * __./POM/__:
        * extracted features: _filler.csv, text_linking_rate.csv, text_synonyms_rate.csv, text_diversity.csv, text_density.csv, text_discourse.csv, text_reference.csv, acoustic.csv, visual.csv_
        * __./labels/__ : directory with extracted separation on classes of low/high dimention
            * .csv files: contain 0/1 classes classification if given transcripts see Section Label Processing for more datails
            * .txt files: contain median of analysed dimention used for separation
     * __./MT/__:
        * extracted features: _~~filler.csv~~, text_linking_rate.csv, text_synonyms_rate.csv, text_diversity.csv, text_density.csv, text_discourse.csv, text_reference.csv, ~~acoustic.csv~~, ~~visual.csv~~_
        * __./labels/__ : directory with extracted separation on classes of low/high dimentions
            * .csv files: contain 0/1 classes classification if given transcripts see Section Label Processing for more datails
            * ~~.txt files: contain median of analysed dimention used for separation~~
        
* __./Models/__: directory contains __ML_Model.py__ with SVM and _Grid Search Cross-Validation_ for searching the best parameters
* __./feedback/__: 
    * __feedback_generator.py__ with function __ABS_SHAP()__ for generation of the _feedback.png_
    * __SHAP.py__ with calculation of _Shapley values_
* __./demo/__: contains saved _background.csv_ and _target.csv_ files for shapley analysis
* __./preprocess/__:
    * __FillerProcessor.py__: extracts filler features outputs _filler.csv_
    * __LabelProcessor.py__: separated POM dimentions on low/high classes outputs _name_of_dimentionLabel.csv_
    * __LabelProcessor_MT.py__: separated MT dimentions on low/high classes outputs _name_of_dimentionLabel.csv_
    * __TextProcessor_MT.py__: extracts textual features outputs _text_linking_rate.csv, text_synonyms_rate.csv, text_diversity.csv, text_density.csv, text_discourse.csv, text_reference.csv_
* __./results/__:
    * __./POM/__: contains folders corresponding to each type of dimention
        ex: __./confident/__: 
            * _F1_score.txt_ with results of average F1 score on 300 experiment with random train/test data splitting;
            * _best_parameters_svm.txt_ containing found by _Models/ML_Model.SupportVectorMachine(X_train, Y_train, X_test, Y_test)_ best parameters for classification with svm;
            * _random_F1_score.txt_ results of random classifier on concidered dimention 
    * __./MT/__: analogous results for _3T_French_ dataset
    

    


## Section Execution

### Subsection: requirements & feature extraction


___ATENTION: some code uses sys.path.append("path\_to\_directory") please make sure that you changed it to your path\_to\_directory in the files SHAP.py LabelProcessor\_MT.py TextProcessor\_MT.py___

Make sure that in your preprocess/utils.py you have:

```
data = pd.read_csv('path_to_your_directory/data/POM/confidenceLabel.csv', index_col=0)
    
```
This code is used only if you extract features from data sets from the scratch which is by the default not mandatory and not recommended as they are provided in this directory. 


First install requirements via:
```
pip install -r requirements.txt
```

___NOTE: IT IS RECOMENDED TO JUST USE FILES WITH EXTRACTED FEATURES & LABELS OF CLASSES THUS ONE CAN SKIP FOLLOWING PART___

If you want to re-extract features you can go to repository __./preprocess/__ and execute code below. To do so make sure that you have directory __.data/name_of_dataset/transcripts/__ containing transcripts of public speech you want to analyse:


```
python3 FillerProcessor.py
python3 TextProcessor_MT.py --dataset MT
```
This will create files: _filler.csv, text_linking_rate.csv, text_synonyms_rate.csv, text_diversity.csv, text_density.csv, text_discourse.csv, text_reference.csv_ in directory __./data/name_of_dataset/__. Once you generated .csv files with extracted features you can reproduce labeling on categories. To do so one can use follwoing commands:

```
python3 LabelProcesssor.py
python3 LabelProcesssor_MT.py
```
 In directory __./data/name_of_dataset/labels/name_of_dimention__ this will create files with separation on low/high classes of respective dimention. 

___ANALYSIS: ONCE LABELING & FEATURE EXTRACTION IS DONE YOU CAN EXECUTE ANALYSIS___


### Subsection: correlation, classification and SHAP analysis



To perform analysis you will have to run _test\_MT.py_ but before please make sure that you changed parameters in the code to what you are interested in: 


```
#______________________ test_MT.py _______________________

# Changeable variables. Please specify data set name: MT or POM 

global dataset
dataset = 'MT'

```

By the default program will analyse _persuasiveness_, _confidense_ dimensions for 3T\_French data set and additionally _engagement_ and _global_ dimensions for POM data set. Make sure that you perform analysis you want by commenting some parts of the main() in _test\_MT.py_ (bolow we specified which parts COULD BE COMMENTED):


```
#______________________ test_MT.py _______________________

if __name__ == "__main__":
    dimentions = getDimentions(dataset)

    for dim in dimentions:
        print("------------------" + dim + "------------------")
        rate_type = dim
        setGlobal(dataset, dim)
        X, group_by_category = loadFeturesByCategories()
        Y = loadRatings()
        X, Y, target, feature_name = dataPreprocessing(X, Y)
        print("*********************** CalculCorr ***********************")
        correlation(X, Y, group_by_category)
        print("*********************** featureSelection ***********************")
        X, Y, target, feature_name = featureSelection(X, Y, target)
        if (len(X.columns) > 0):
            print("*********************** Chose Best Parameters ***********************")
            best_param, best_model = choseBestParametersForClassification(X, Y)
            
# If you are not interested in  LeaveOneOut evaluation please comment this part:
            print("*********************** LeaveOneOut ***********************")
            leaveOneOutTrain(X, Y, best_param, best_model)
            
# If you are not interested in checking the evaluation of performance of the Random classifier please comment this part:
            print("*********************** Random ***********************")
            randomClassifier(X, Y)
            
# If you are not interested in calculation of SHAP values please comment this part:
            print("*********************** SHAP ***********************")
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
            best_model.fit(X_train, y_train.values.ravel())
            shapAnalysis(best_model, X_train, X_test, target, group_by_category, feature_name)

```



We already saved textual, filler and audio categories in file _categories.json_. To train classifier and analyse the performance you can execute following:

```
python3 test_MT.py
```

If you want to change the set of used features go to the function _loadFeturesByCategories()_ and change construction of variable X. By the default program is concidering all linking_rate, synonym_rate, diversity, density, discourse, reference, polarity features for 3T\_French dataset and additionally to this list filler features for POM data set. To change that following configurations could be used:

```
# Use linking_rate, synonym_rate, diversity, density, discourse, reference, polarity features:

X = text_linking_rate.join(text_synonym_rate, how = 'left').join(text_div, how = 'left').join(text_dens, how = 'left').join(text_disc, how = 'left').join(text_ref, how = 'left') 

# To add fillers:

X = text_linking_rate.join(text_synonym_rate, how = 'left').join(text_div, how = 'left').join(text_dens, how = 'left').join(text_disc, how = 'left').join(text_ref, how = 'left').join(filler, how = 'left')

```


Execution of __test_MT.py__ will create files in __./results/__ and __./demo/__ described before. 


## Section Feature extraction 

In this section we will provide details on how textual features were extracted. Plase for more details check corresponding code in the directory __./preprocessing/__. For more information on diversity, density, reference, overlapping features please check \url{http://arxiv.org/abs/1612.00729}. For more information on fillers see \url{https://inria.hal.science/hal-02933476/document}.

### Lexical Diversity Features



$TTR = \dfrac{nb\_types}{nb\_tokens}$ \
$CorrectedTTR = \dfrac{nb\_types}{\sqrt{2*nb\_tokens}}$\
$RootTTR = \dfrac{nb\_types}{\sqrt{nb\_tokens}}$\
$BilogTTR = \dfrac{log(nb\_types)}{log(nb\_tokens)}$\
MTLD measures the average length of continuous text sequence that maintains the TTR above a threshold of $0.72$
  
### Density Features 


Taking into account:


$numLexicals=numAdj+numNouns+numVerbs+numAdverbs+numProperNouns$
$numVerbsOnly=numVerbs$


Density features are calculated as:
$POS_numNouns=\dfrac{numNouns+numProperNouns}{TotalWords}$\
$POS_numProperNouns=\dfrac{numProperNouns}{TotalWords}$\
$POS_numPronouns=\dfrac{numPronouns}{TotalWords}$  \
$POS_numConjunct=\dfrac{numConjunct}{TotalWords}$\
$POS_numAdjectives=\dfrac{numAdj}{TotalWords}$ \
$POS_numVerbs=\dfrac{numVerbs}{TotalWords}$ \
$POS_numAdverbs=\dfrac{numAdverbs}{TotalWords}$ \
$POS_numPrepositions=\dfrac{numPrepositions}{TotalWords}$ \
$POS_numInterjections=\dfrac{numInterjections}{TotalWords}$ \
$POS_numPerPronouns=\dfrac{perpronouns}{TotalWords}$ \
$POS_numLexicals=\dfrac{numLexicals}{TotalWords}$ \
$POS_numFunctionWords=\dfrac{numFunctionWords}{TotalWords}$ \
$POS_numDeterminers=\dfrac{numDeterminers}{TotalWords}$ \
$POS_numVerbsVB=\dfrac{numVB}{TotalWords}$ \
$POS_numVerbsVBN=\dfrac{numVBN}{TotalWords}$ \
$POS_advVar=\dfrac{numAdverbs}{numLexicals}$ \
$POS_adjVar=\dfrac{numAdj}{numLexicals}$ \
$POS_modVar=\dfrac{numAdj+numAdverbs}{numLexicals}$ \
$POS_nounVar=\dfrac{numNouns+numProperNouns}{numLexicals}$ \
$POS_verbVar1=\dfrac{numVerbsOnly}{len(uniqueVerbs)}$ \
$POS_verbVar2=\dfrac{numVerbsOnly}{numLexicals}$ \
$POS_squaredVerbVar1=\dfrac{numVerbsOnly\times numVerbsOnly}{len(uniqueVerbs)}$\
$POS_correctedVV1=\dfrac{numVerbsOnly}{\sqrt{2.0\times len(uniqueVerbs)}}$ 
    
    
    
### Reference Features


$numPronouns=numPersonalPronouns+numPossessivePronouns$\
$DISC_RefExprPronounsPerNoun=\dfrac{numPronouns}{numNouns}$ \
$DISC_RefExprPronounsPerSen=\dfrac{numPronouns}{numSentences}$ \
$DISC_RefExprPronounsPerWord=\dfrac{numPronouns}{numWords}$ \
$DISC_RefExprPerPronounsPerSen=\dfrac{numPersonalPronouns}{numSentences}$ \
$DISC_RefExprPerProPerWord=\dfrac{numPersonalPronouns}{numWords}$ \
$DISC_RefExprPossProPerSen=\dfrac{numPossessivePronouns}{numSentences}$ \
$DISC_RefExprPossProPerWord=\dfrac{numPossessivePronouns}{numWords}$ \
$DISC_RefExprDefArtPerSen=\dfrac{numDefiniteArticles}{numSentences}$ \
$DISC_RefExprDefArtPerWord=\dfrac{numDefiniteArticles}{numWords}$ \
$DISC_RefExprProperNounsPerNoun=\dfrac{numProperNouns}{numNouns}$ 

### Overlap Features



totalSentencesSize -- number of sentences within the transcript.

_localNounOverlapCount_ measures wheather there is the same noun withing two subsequent sentences. _localStemOverlapCount_ measures wheather there is the same noun withing two subsequent sentences if not then if there is either noun or pronoun with the same lemma. _localArgumentOverlapCount_ measures wheather there is the same noun withing two subsequent sentences if not then if there is exact the same pronoun within two subsequent sentences and if not then if there is any same lemma for any noun of pronoun. _localContentWordOverlap_ measures how many words (not pronouns) with same lemmas there are in two subsequent sentences. 

Analogously _global_ features measures same quantities but for any two sentences. Finally all listed features are normalised by the devision by total number of sentences within transcript. 


### Linking Rate Features

$conjunctToSent=\dfrac{conjunctNum}{nbSent}$ \
$conjunctTypesToSent=\dfrac{numTypesConjunt}{nbSent}$ \
$conjunctToWords=\dfrac{conjunctNum}{docLen}$ \
$conjunctNeighborSent=\dfrac{numSameConjunctForSubsequentSentences}{nbSent}$ 


### Synonym Rate Features

$synonymToNouns=\dfrac{nbOfGroupsOfNounSynonyms}{nbOfNouns}$ \
averageSynClassNOUN average number of used synonyms within synonym classes for nouns \
$synonymToVerbs=\dfrac{nbOfGroupsVerbSynonyms}{nbOfVerbs}$ \
averageSynClassVERB average number of used synonyms within synonym classes for verbs 

### Filler Features

f_uh number of fillers "uhh"
f_um number of fillers "umm"
f_start number of fillers at the start of the sentence
f_mid number of fillers within the sentence
f_uncertain number of fillers in sentences containing stutter
f_sen average number of tokens within the sentences containing fillers

### Polarity Features

Those features were extracted using LIWC tool fron \url{https://www.liwc.app}.

posemo: percentages of positive words
negemo: percentages of negative words
anx: percentages of anxious words
anger: percentages of angry words
sad:percentages of sad words

## Lexical Features Overview

### "Language Level"

## Lexical Diversity

| **Feature**       | **Formula**                                          | **Description**                                                                                                                                               |
|-------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Traditional TTR (TTR) | $$\dfrac{\text{Types}}{\text{Tokens}}$$             | Measures the ratio of unique words (types) to the total number of words (tokens), indicating lexical variety.                                                   |
| Corrected TTR     | $$\dfrac{\text{Types}}{\sqrt{2 \times \text{Tokens}}}$$ | Adjusts TTR for text length, providing a more stable estimate of lexical diversity.                                                                             |
| Root TTR          | $$\dfrac{\text{Types}}{\sqrt{\text{Tokens}}}$$          | Another length-corrected measure of lexical diversity, less sensitive to text length than traditional TTR.                                                      |
| Bilog TTR         | $$\dfrac{\log(\text{Types})}{\log(\text{Tokens})}$$     | Logarithmic transformation of TTR, providing a normalized measure of lexical richness.                                                                         |
| MTLD              | $$\mu(\text{TTR}(\text{text}) > 0.72)$$                | Calculates the average sequence length where TTR remains above 0.72, a threshold indicating sustained lexical diversity.                                        |

#### POStag Density

| **Notation**            | **Formula**                                               | **Description**                                                                                               |
|-------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| NounDens                | \( \dfrac{\text{numNouns} + \text{numProperNouns}}{\text{TotalWords}} \) | Ratio of nouns and proper nouns to total words, indicating noun density of the text.                          |
| ProNounDens             | \( \dfrac{\text{numProperNouns}}{\text{TotalWords}} \)     | Proportion of proper nouns, reflecting the usage of specific entities or names.                               |
| PronDens                | \( \dfrac{\text{numPronouns}}{\text{TotalWords}} \)        | Measures the frequency of pronouns, suggesting narrative perspective or anonymity.                            |
| ConjDens                | \( \dfrac{\text{numConjunct}}{\text{TotalWords}} \)        | Proportion of conjunctions, indicating the complexity or compound structure of sentences.                     |
| AdjDens                 | \( \dfrac{\text{numAdj}}{\text{TotalWords}} \)             | Ratio of adjectives, reflecting descriptive intensity or detail in the text.                                  |
| VerbDens                | \( \dfrac{\text{numVerbs}}{\text{TotalWords}} \)           | Proportion of verbs, indicating action or dynamic content in the text.                                        |
| AdvDens                 | \( \dfrac{\text{numAdverbs}}{\text{TotalWords}} \)         | Measures the adverb usage, giving insights into the modification of verbs, adjectives, or other adverbs.       |
| PreposDens              | \( \dfrac{\text{numPrepositions}}{\text{TotalWords}} \)    | Frequency of prepositions, pointing to relational dynamics between text elements.                             |
| InterjDens              | \( \dfrac{\text{numInterjections}}{\text{TotalWords}} \)   | Ratio of interjections, showing emotional or colloquial speech inclusions.                                    |
| PersPronDens            | \( \dfrac{\text{perpronouns}}{\text{TotalWords}} \)        | Proportion of personal pronouns, potentially highlighting narrative involvement or focus.                     |
| LexDens                 | \( \dfrac{\text{numLexicals}}{\text{TotalWords}} \)        | Measures lexical content using a composite of various word types, indicating textual richness.                 |
| FuncWordDens            | \( \dfrac{\text{numFunctionWords}}{\text{TotalWords}} \)   | Ratio of function words, essential for grammatical structure but low in content value.                        |
| DeterDens               | \( \dfrac{\text{numDeterminers}}{\text{TotalWords}} \)     | Frequency of determiners, important for noun phrase specification and clarity.                                |
| BaseFormVerbDen         | \( \dfrac{\text{numVB}}{\text{TotalWords}} \)              | Ratio of base form verbs (infinitive), indicating potential for statements of general truths or imperatives.   |
| PastParticipVerbDens    | \( \dfrac{\text{numVBN}}{\text{TotalWords}} \)             | Proportion of past participle verbs, often reflective of passive constructions or perfect aspects.            |
| AdvVar                  | \( \dfrac{numAdverbs}{numLexicals} \)                      | Measures the proportion of adverbs relative to the total number of lexical words, indicating adverb diversity. |
| AdjVar                  | \( \dfrac{numAdj}{numLexicals} \)                          | Indicates the ratio of adjectives to lexical words, reflecting richness of adjective usage in the text.        |
| ModVar                  | \( \dfrac{numAdj+numAdverbs}{numLexicals} \)               | Reflects the diversity of both adjectives and adverbs relative to lexical words.                              |
| NounVar                 | \( \dfrac{numNouns+numProperNouns}{numLexicals} \)         | Proportion of nouns relative to lexical words, indicating noun diversity.                                     |
| VerbVar1                | \( \dfrac{numVerbsOnly}{len(uniqueVerbs)} \)               | Measures verb diversity as the ratio of unique verbs to total verbs used.                                     |
| VerbVar2                | \( \dfrac{numVerbsOnly}{numLexicals} \)                    | Reflects diversity of verbs relative to lexical words, considering all verb occurrences.                      |
| SqVerbVar               | \( \dfrac{numVerbsOnly \times numVerbsOnly}{len(uniqueVerbs)} \) | Provides a measure of verb diversity by considering the square of total verbs to unique verbs.                |
| CorrVerbVar             | \( \dfrac{\text{numVerbsOnly}}{\sqrt{2.0 \times len(\text{uniqueVerbs})}} \) | Adjusts verb variation for text length, providing a more stable estimate of verb diversity.                    |

#### Overlapping

| **Notation**        | **Description**                                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------------------------------|
| TotalSentSize       | Total number of sentences in the transcript, indicating overall length and complexity.                           |
| LocNounOverlap      | Measures the occurrence of the same noun in two consecutive sentences, indicating noun repetition and topic continuity. |
| LocStemOverlap      | Counts occurrences of the same nouns or pronouns with the same lemma in two consecutive sentences, reflecting lexical or thematic consistency. |
| LocArgOverlap       | Measures the repetition of nouns or exact pronouns, or any similar lemma across two consecutive sentences, indicating argumentative or narrative linkage. |
| LocContWordOverlap  | Tracks the number of non-pronoun words with the same lemmas appearing in two consecutive sentences, showing depth of content repetition. |
| **Global Features** | Measures the same quantities for any two sentences, with all features normalized by division by the total number of sentences in the transcript. |


#### Referential

| **Notation**         | **Formula**                                              | **Description**                                                                                               |
|----------------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| PronPerNoun          | \( \dfrac{\text{numPronouns}}{\text{numNouns}} \)        | Ratio of total pronouns to nouns, indicating pronoun usage dependency on nouns in the text.                   |
| PronPerSent          | \( \dfrac{\text{numPronouns}}{\text{numSentences}} \)    | Average number of pronouns used per sentence, suggesting narrative focus or subject continuity.                |
| PronPerWord          | \( \dfrac{\text{numPronouns}}{\text{numWords}} \)        | Frequency of pronouns relative to total word count, highlighting pronoun density.                              |
| PersPronPerSent      | \( \dfrac{\text{numPersonalPronouns}}{\text{numSentences}} \) | Frequency of personal pronouns per sentence, reflecting personal narrative or direct address.                 |
| PersPronPerWord      | \( \dfrac{\text{numPersonalPronouns}}{\text{numWords}} \) | Density of personal pronouns relative to word count, indicating personal engagement.                          |
| PossPronPerSent      | \( \dfrac{\text{numPossessivePronouns}}{\text{numSentences}} \) | Proportion of possessive pronouns per sentence, showing possession or belonging expressed per sentence.       |
| PossPronPerWord      | \( \dfrac{\text{numPossessivePronouns}}{\text{numWords}} \) | Frequency of possessive pronouns against total words, indicating possession or relational dynamics.            |
| DefArtPerSent        | \( \dfrac{\text{numDefiniteArticles}}{\text{numSentences}} \) | Frequency of definite articles per sentence, suggesting specificity or focus on known entities.               |
| DefArtPerWord        | \( \dfrac{\text{numDefiniteArticles}}{\text{numWords}} \) | Proportion of definite articles relative to total words, indicating textual specificity.                       |
| ProNounPerNoun       | \( \dfrac{\text{numProperNouns}}{\text{numNouns}} \)     | Ratio of proper nouns to common nouns, indicating prevalence of named entities versus general terms.           |

#### SD LIWC

| **Feature**      | **Description**                                                                                               |
|------------------|---------------------------------------------------------------------------------------------------------------|
| WordPerSent      | Average number of words per sentence, providing a measure of sentence length and complexity.                   |
| WordCount        | Total count of words in the text, indicating overall text length.                                              |
| BigWords         | Number of words longer than six letters, reflecting the presence of longer and potentially more complex words. |


#### Synonyms

| **Notation**          | **Formula**                                              | **Description**                                                                                               |
|-----------------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| SynNounsRatio         | \( \dfrac{\text{nbOfGroupsOfNounSynonyms}}{\text{nbOfNouns}} \) | Measures the proportion of noun synonym groups relative to the total number of nouns, indicating lexical diversity. |
| AverSynPerNounClass   | \( \text{AVER}(\text{SynClassNOUNSize}) \)               | Represents the average number of synonyms used within each noun synonym class, reflecting noun variability.     |
| SynToVerbRatio        | \( \dfrac{\text{nbOfGroupsVerbSynonyms}}{\text{nbOfVerbs}} \) | Measures the proportion of verb synonym groups relative to the total number of verbs, indicating lexical diversity. |
| AverSynPerVerbClass   | \( \text{averageSynClassVERB} \)                         | Represents the average number of synonyms used within each verb synonym class, reflecting verb variability.      |

#### Transitions

| **Notation**            | **Formula**                                              | **Description**                                                                                               |
|-------------------------|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| ConjToSentRatio          | \( \dfrac{\text{conjunctNum}}{\text{nbSent}} \)         | Measures the frequency of conjunctions per sentence, indicating sentence complexity and compound structure.    |
| ConjTypesToSentRatio     | \( \dfrac{\text{numTypesConjunt}}{\text{nbSent}} \)     | Assesses the variety of conjunction types used per sentence, reflecting syntactic diversity and complexity.    |
| ConjToWordsRatio         | \( \dfrac{\text{conjunctNum}}{\text{docLen}} \)         | Ratio of conjunctions to the total number of words, indicating the use of conjunctions to link ideas or clauses. |
| ConjNeighborSent         | \( \dfrac{\text{numSameConjunctForSubsequentSentences}}{\text{nbSent}} \) | Measures the frequency of the same conjunction used in consecutive sentences, indicating cohesive use of conjunctions. |


#### Affective Cognitive and Perceptive Processes
##### Affective

| **Feature**    | **Description**                                                                                               |
|----------------|---------------------------------------------------------------------------------------------------------------|
| PosEmo         | Percentage of words conveying positive emotions, indicating positive sentiment or attitude.                    |
| NegEmo         | Percentage of words conveying negative emotions, indicating negative sentiment or attitude.                    |
| Anx            | Percentage of words expressing anxiety, reflecting emotional stress or tension.                                |
| Anger          | Percentage of words expressing anger, indicating hostility or frustration.                                     |
| Sad            | Percentage of words conveying sadness, indicating depressive or sorrowful moods.                               |

##### Cognitive

| **Feature**     | **Description**                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| Cognition       | Percentage of words related to cognitive processes, reflecting thought complexity and depth.                   |
| Insight         | Percentage of words that suggest insights, reflecting reflective or learning processes.                        |
| Cause           | Percentage of words indicating causation, used to describe reasons or explanations.                            |
| Divergence      | Percentage of words indicating divergence, suggesting difference or deviation.                                 |
| Tentative       | Percentage of words expressing tentativeness, indicating uncertainty or hesitancy.                             |
| Certainty       | Percentage of words expressing certainty, indicating assertiveness or confidence.                              |
| Inhibition      | Percentage of words suggesting restraint or restriction in behavior or expression.                             |
| Inclusion       | Percentage of words implying inclusion, indicating collective involvement or engagement.                       |
| Exclusion       | Percentage of words implying exclusion, suggesting separation or detachment.                                   |

##### Perceptive

| **Feature**    | **Description**                                                                                               |
|----------------|---------------------------------------------------------------------------------------------------------------|
| Perception     | Percentage of words related to sensory or perceptual processes, reflecting engagement with the environment.     |

