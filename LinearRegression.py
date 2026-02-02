import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Input feature (X) and output variable (y)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)
# Predict output values
y_pred = model.predict(X)

# Predict for a new input
new_value = np.array([[6]])
prediction = model.predict(new_value)

print("Predicted value for input 6:", prediction[0])
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Input Variable (X)")
plt.ylabel("Output Variable (y)")
plt.title("Linear Regression Example")
plt.show()
