import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('diabetes.csv')
print(data.head())# Define the features and target variable
X = data.drop('Outcome', axis=1)    
y = data['Outcome']
# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)
# Make predictions
predictions = model.predict(X)
print(predictions)