import pandas as pd


def confidenceKeys():
    data = pd.read_csv('/Users/goddessoffailures/Desktop/Mock-Up_sample/Mock-Up_sample/REVITALISE/data/POM/confidenceLabel.csv', index_col=0)
    return list(data.index)

