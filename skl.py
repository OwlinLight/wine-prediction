import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC


# Load the data
data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')


wine = data
# trim
cols_to_filter = [col for col in wine.columns if col not in ['Id', 'density', 'pH', 'quality']]  # Define the columns to filter
q3 = wine[cols_to_filter].quantile(0.75)  # Calculate the third quartile of each included column
q1 = wine[cols_to_filter].quantile(0.25)  # Calculate the third quartile of each included column
mask = (wine[cols_to_filter] <= q3+1.5*(q3-q1)).all(axis=1)  # Create a mask that selects only the rows where included columns are less than or equal to Q3
wine = wine[mask]  # Remove rows where included columns are greater than Q3

y = wine['quality']
X = wine.drop(['quality', 'Id'], axis = 1)


# Define the number of folds for cross-validation
n_folds = 5
# Create a KFold object to split the dataset into training and testing folds
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
# Initialize an empty list to store the accuracies
accuracies = []

# ada_boost = SVC(C=1.2, gamma=0.9, kernel='rbf')
from sklearn.ensemble import GradientBoostingRegressor
reg1 = GradientBoostingRegressor(random_state=1)

from sklearn.ensemble import RandomForestRegressor
reg2 = RandomForestRegressor(random_state=1)

from sklearn.linear_model import LinearRegression
reg3 = LinearRegression()

from sklearn.ensemble import VotingRegressor
regressor = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lir', reg3)])

ada_boost = regressor

# ada_boost = RandomForestClassifier(random_state = 42)
scaler = StandardScaler()

X = X.to_numpy()
y = y.to_numpy()

os=SMOTE()
X, y = os.fit_resample(X, y)

# Loop over the folds and train/test the classifier on each fold
for fold, (train_indices, test_indices) in enumerate(kfold.split(X, y)):
    # Get the training and testing data for this fold
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Train the classifier on the training set
    ada_boost.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = ada_boost.predict(X_test).astype(int)

    # Calculate the accuracy of the classifier for this fold
    accuracy = accuracy_score(y_test, y_pred)
    # accuracy = 1

    # Print the accuracy for this fold
    print("Fold {}: Accuracy: {:.2f}%".format(fold + 1, accuracy * 100))

    # Add the accuracy to the list of accuracies
    accuracies.append(accuracy)

# Calculate the average accuracy over all folds
avg_accuracy = sum(accuracies) / len(accuracies)

# Print the average accuracy
print("Average accuracy: {:.2f}%".format(avg_accuracy * 100))


y_id = test_data['Id']
X_test = test_data.drop(['Id'], axis=1)

X_test = scaler.fit_transform(X_test)
y_pred = ada_boost.predict(X_test)
y_pred = np.rint(y_pred).astype(int)

c1 = y_id.to_frame()
c2 = pd.DataFrame({'quality': y_pred})
df = pd.concat([c1, c2], axis = 1)
df.to_csv('output.csv', index=False)

