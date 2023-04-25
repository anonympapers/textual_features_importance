
# import java.util.List;
# import java.util.TreeMap;
#
# import edu.stanford.nlp.ling.TaggedWord;
# import utils.genutils.NumUtils;

listOfPOSTagFeatures = ['POS_numNouns', 'POS_numProperNouns', 'POS_numPronouns', 'POS_numConjunct', 'POS_numAdjectives', 'POS_numVerbs',
                        'POS_numAdverbs',  'POS_numModals', 'POS_numPrepositions',  'POS_numInterjections', 'POS_numPerPronouns',
                        'POS_numWhPronouns', 'POS_numLexicals', 'POS_numFunctionWords', 'POS_numDeterminers',
                        'POS_numVerbsVB', 'POS_numVerbsVBG', 'POS_numVerbsVBN', 'POS_numVerbsVBP', 'POS_numVerbsVBZ',
                        'POS_advVar', 'POS_adjVar', 'POS_modVar', 'POS_nounVar', 'POS_verbVar1', 'POS_verbVar2', 'POS_squaredVerbVar1',
                        'POS_correctedVV1']

posFeatures = dict.fromkeys(listOfPOSTagFeatures,0)

posFeatures["POS_numNouns"]= (['numNouns'] + ['numProperNouns']) / ['TotalWords']
posFeatures["POS_numProperNouns"] = ['numProperNouns'] / ['TotalWords']
posFeatures["POS_numPronouns"] = ['numPronouns'] / ['TotalWords']
posFeatures["POS_numConjunct"] = ['numConjunct'] / ['TotalWords']
posFeatures["POS_numAdjectives"] = ['numAdj'] / TotalWords
posFeatures["POS_numVerbs"] = ['numVerbs'] / TotalWords));
posFeatures["POS_numAdverbs"] = ['numAdverbs'] / TotalWords));
posFeatures["POS_numModals"] =  ['numModals'] / TotalWords));
posFeatures["POS_numPrepositions"] = ['numPrepositions'] / TotalWords));
posFeatures["POS_numInterjections"] = ['numInterjections'] / TotalWords));
posFeatures["POS_numPerPronouns"] = ['perpronouns'] / TotalWords));
posFeatures["POS_numWhPronouns"] = ['whperpronouns'] / TotalWords));
posFeatures["POS_numLexicals"] = ['numLexicals'] / TotalWords)); #Lexical Density
posFeatures["POS_numFunctionWords"] = ['numFunctionWords'] / TotalWords));
posFeatures["POS_numDeterminers"] = ['numDeterminers'] / TotalWords));
posFeatures["POS_numVerbsVB"] = ['numVB'] / TotalWords));
posFeatures["POS_numVerbsVBD"] = ['numVBD'] / TotalWords));
posFeatures["POS_numVerbsVBG" =  ['numVBG'] / TotalWords));
posFeatures["POS_numVerbsVBN"] =  ['numVBN'] / TotalWords));
posFeatures["POS_numVerbsVBP"] = ['numVBP'] / TotalWords));
posFeatures["POS_numVerbsVBZ"] = ['numVBZ'] / TotalWords));
posFeatures["POS_advVar"] = ['numAdverbs'] / ['numLexicals']));
posFeatures["POS_adjVar"] = ['numAdj'] / numLexicals));
posFeatures["POS_modVar"] = (['numAdj'] + ['numAdverbs']) / numLexicals));
posFeatures["POS_nounVar"] = (['numNouns'] + ['numProperNouns']) / numLexicals));
posFeatures["POS_verbVar1"] = ['numVerbsOnly'] / uniqueVerbs.size())); # VV1
posFeatures["POS_verbVar2"] = ['numVerbsOnly'] / numLexicals)); # VV2
posFeatures["POS_squaredVerbVar1"] = (['numVerbsOnly'] * ['numVerbsOnly']) / uniqueVerbs.size())); # VV1
posFeatures["POS_correctedVV1"] = ['numVerbsOnly'] / math.sqrt(2.0 * uniqueVerbs.size()))); # CVV1
