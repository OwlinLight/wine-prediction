import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

# Split the data into training and testing sets
X = data.drop(['quality', 'Id'], axis=1)
y = data['quality']

# X_train, y_train = X, y
# X_test = test_data.drop(['Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train.values)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test.values)

# Define the neural network model
class WineNet(nn.Module):
    def __init__(self):
        super(WineNet, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the model
model = WineNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train.reshape(y_train.shape[0], 1))
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_test.reshape(y_test.shape[0], 1))
        print(f"Epoch {epoch}: Mean squared error: {test_loss:.2f}")

#
# # Evaluate the model on the test set
with torch.no_grad():
    test_output = model(X_test)
    # test_loss = criterion(test_output, y_test)
    test_loss = criterion(test_output, y_test.reshape(y_test.shape[0], 1))
print("Mean squared error: {:.2f}".format(test_loss))

with torch.no_grad():
    output = model(X_test)
    df = pd.DataFrame(output.numpy())

df.to_csv('output.csv', index=False)



# Make predictions on new data
new_data = scaler.transform([[7.2, 0.35, 0.36, 1.8, 0.072, 28, 66, 0.994, 3.23, 0.73, 9.7]])
new_data_tensor = torch.Tensor(new_data)
prediction = model(new_data_tensor)
print("Predicted wine quality: {:.1f}".format(prediction.item()))