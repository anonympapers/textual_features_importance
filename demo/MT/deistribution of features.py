import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

global dataset
dataset = 'MT'

global rate_type
rate_type = 'confidence'

# Download MT180 ratings
features = pd.read_csv( "./" + rate_type + "/background.csv", sep = ",", index_col = 0 , dtype = 'string')
print(features)
columns = features.columns
index= features.index

#Converting feature values to float64
cutted = np.array(features[features.columns[0:]])
cutted = pd.DataFrame(cutted, index = index, columns = columns[0:], dtype = np.float64)
columns_cutted = cutted.columns
features[columns_cutted] = cutted[columns_cutted]
print(features)
print(columns)

target_features = ['globalContentWordOverlap', 'conjunctNum', 'MTLD', 'POS_squaredVerbVar1']

fig, axes = plt.subplots(nrows = 2, ncols =2, figsize = (8, 6))
colors = ['olive', 'olive', 'olive', 'olive',  "#e24a33", "#348abd", "#988ed5", "#777777", 'red', 'pink', 'yellow',  "#e24a33", "#348abd", "#988ed5", "#777777",] # whatever the colors may be but it should be different for each histogram.

for index, name in enumerate(target_features):
    ax = axes.flatten()[index]
    ax.hist(features[name], color = colors[index], label = name)
    ax.legend(loc = "best")
    ax.grid(visible = True)
# plt.suptitle("Histograms of DS characteristics", size = 20)
plt.show()
