import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC

# Load the data
data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

wine = data
cols_to_filter = [col for col in wine.columns if col not in ['id', 'density', 'pH', 'quality']]  # Define the columns to filter
q3 = wine[cols_to_filter].quantile(0.75)  # Calculate the third quartile of each included column
q1 = wine[cols_to_filter].quantile(0.25)  # Calculate the third quartile of each included column
mask = (wine[cols_to_filter] <= q3+1.5*(q3-q1)).all(axis=1)  # Create a mask that selects only the rows where included columns are less than or equal to Q3
wine = wine[mask]  # Remove rows where included columns are greater than Q3

# Split the data into training and testing sets
X = wine.drop(['quality', 'Id'], axis=1)
y = wine['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# wp = WineDataProcessor(X_train, X_test)
# X_train, X_test = wp.process_data()

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


ada_boost = SVC(C=1.2, gamma=0.9, kernel='rbf')
# ada_boost = SVC(C=50,kernel="rbf")

# ada_boost = AdaBoostClassifier(n_estimators=50, random_state=43)

# Train the classifier on the training set
ada_boost.fit(X_train, y_train)

# # Make predictions on the testing set
y_pred = ada_boost.predict(X_test)
df = pd.DataFrame(y_pred)
df.to_csv('output.csv', index=False)

# Calculate the accuracy of the classifier
accuracy = ada_boost.score(X_test, y_test)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
