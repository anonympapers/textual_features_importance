# generate feature groups of each behaviour
import json
audio_category = {'prosody': ['speech_rate_words', 'articulation_words', 'pause_rate', 'pause_ratio', 'pause_mean_dur', 'pause_perc']
                             + ['F0semitoneFrom27.5Hz_sma3nz_amean','F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
                                'F0semitoneFrom27.5Hz_sma3nz_percentile20.0','F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
                                'F0semitoneFrom27.5Hz_sma3nz_percentile80.0','F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
                                'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope','F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
                                'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope','F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
                                'loudness_sma3_amean', 'loudness_sma3_stddevNorm', 'loudness_sma3_percentile20.0','loudness_sma3_percentile50.0',
                                'loudness_sma3_percentile80.0', 'loudness_sma3_pctlrange0-2','loudness_sma3_meanRisingSlope',
                                'loudness_sma3_stddevRisingSlope', 'loudness_sma3_meanFallingSlope','loudness_sma3_stddevFallingSlope',
                                'loudnessPeaksPerSec',
                                ],
                  'voice_quality': ['VoicedSegmentsPerSec',
                                    'MeanVoicedSegmentLengthSec',
                                    'StddevVoicedSegmentLengthSec',
                                    'MeanUnvoicedSegmentLength',
                                    'StddevUnvoicedSegmentLength',
                                    'jitterLocal_sma3nz_amean',
                                    'jitterLocal_sma3nz_stddevNorm',
                                    'shimmerLocaldB_sma3nz_amean',
                                    'shimmerLocaldB_sma3nz_stddevNorm',
                                    'HNRdBACF_sma3nz_amean',
                                    'HNRdBACF_sma3nz_stddevNorm',
                                    'logRelF0-H1-H2_sma3nz_amean',
                                    'logRelF0-H1-H2_sma3nz_stddevNorm',
                                    'logRelF0-H1-A3_sma3nz_amean',
                                    'logRelF0-H1-A3_sma3nz_stddevNorm',
                                    'F1frequency_sma3nz_amean',
                                    'F1frequency_sma3nz_stddevNorm',
                                    'F1bandwidth_sma3nz_amean',
                                    'F1bandwidth_sma3nz_stddevNorm',
                                    'F1amplitudeLogRelF0_sma3nz_amean',
                                    'F1amplitudeLogRelF0_sma3nz_stddevNorm',
                                    'F2frequency_sma3nz_amean',
                                    'F2frequency_sma3nz_stddevNorm',
                                    'F2bandwidth_sma3nz_amean',
                                    'F2bandwidth_sma3nz_stddevNorm',
                                    'F2amplitudeLogRelF0_sma3nz_amean',
                                    'F2amplitudeLogRelF0_sma3nz_stddevNorm',
                                    'F3frequency_sma3nz_amean',
                                    'F3frequency_sma3nz_stddevNorm',
                                    'F3bandwidth_sma3nz_amean',
                                    'F3bandwidth_sma3nz_stddevNorm',
                                    'F3amplitudeLogRelF0_sma3nz_amean',
                                    'F3amplitudeLogRelF0_sma3nz_stddevNorm',
                                    'alphaRatioV_sma3nz_amean',
                                    'alphaRatioV_sma3nz_stddevNorm',
                                    'hammarbergIndexV_sma3nz_amean',
                                    'hammarbergIndexV_sma3nz_stddevNorm',
                                    'slopeV0-500_sma3nz_amean',
                                    'slopeV0-500_sma3nz_stddevNorm',
                                    'slopeV500-1500_sma3nz_amean',
                                    'slopeV500-1500_sma3nz_stddevNorm',
                                    'alphaRatioUV_sma3nz_amean',
                                    'hammarbergIndexUV_sma3nz_amean',
                                    'slopeUV0-500_sma3nz_amean',
                                    'slopeUV500-1500_sma3nz_amean',
                                    'equivalentSoundLevel_dBp'
                                    ]
                  }

# print(len(audio_category['prosody']))
# print(len(audio_category['voice_quality']))
from preprocess.VisualProcessor import au_presence_n, au_intensity_n
visual_category = {'facial_expression':au_presence_n + au_intensity_n}

filler_category = {'fluency':['f_uh', 'f_um', 'f_start', 'f_mid', 'f_uncertain', 'sen_len']}
text_category = {'linking_rate': ['linking_rate'], 'synonyms_rate':['synonyms_rate'], 'embedding':['emb_axis_' + str(i) for i in range(100)]}

print(text_category)
categories = {'audio_category': audio_category,'visual_category':visual_category,'filler_category':filler_category, 'text_category': text_category}

with open('./categories.json', "w") as f:
    json.dump(categories, f)