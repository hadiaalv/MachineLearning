from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  #Hours studied
y = np.array([1, 3, 2, 3, 5])            #Scores

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict score for 6 hours of study
print("Predicted Score:", model.predict([[6]]))
