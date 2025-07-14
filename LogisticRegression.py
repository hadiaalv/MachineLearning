from sklearn.linear_model import LogisticRegression

# Features: [Hours Studied, Slept Well]
X = [[2, 1], [4, 1], [6, 0], [8, 1]]
y = [0, 0, 1, 1]  # 0 = Fail, 1 = Pass

model = LogisticRegression()
model.fit(X, y)

# Predict
print("Will pass:", model.predict([[5, 1]]))  # Output: 0 or 1
