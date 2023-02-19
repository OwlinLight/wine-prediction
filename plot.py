import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

# Split the data into training and testing sets
X = data.drop(['quality', 'Id'], axis=1)
y = data['quality']

# # plot the corrolation graph
# correlation=data.corr()
# plt.figure(figsize=(10,10))
# sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',cmap='Reds',annot=True,annot_kws={'size':8})
# plt.show()

wine = data
# cols_to_filter = [col for col in wine.columns if col not in ['id', 'density', 'pH', 'quality']]  # Define the columns to filter
# q3 = wine[cols_to_filter].quantile(0.75)  # Calculate the third quartile of each included column
# q1 = wine[cols_to_filter].quantile(0.25)  # Calculate the third quartile of each included column
# mask = (wine[cols_to_filter] <= q3+1.5*(q3-q1)).all(axis=1)  # Create a mask that selects only the rows where included columns are less than or equal to Q3
# wine = wine[mask]  # Remove rows where included columns are greater than Q3



# # figure the data


# Create a figure with 13 subplots (one for each feature)
fig, axs = plt.subplots(13, 1, figsize=(10, 20))

# Loop over the features and plot the distribution of each feature
for i, col in enumerate(wine.columns):
    # Get the data for this feature
    data = wine[col]
    # Plot the distribution using a histogram
    sns.boxplot(data, ax=axs[i], orient='h')
    # Set the title of the subplot to the name of the feature
    axs[i].set_title(col)
# Adjust the spacing between the subplots
fig.tight_layout()

# Show the plot
plt.show()


