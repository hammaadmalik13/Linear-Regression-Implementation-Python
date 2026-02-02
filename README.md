# Linear-Regression-Implementation-Python
A simple Python implementation of Linear Regression using NumPy, Matplotlib, and scikit-learn to train a model, make predictions, and visualize the regression line.

# Linear Regression Implementation in Python

This repository contains a Python program that demonstrates **Linear Regression**, one of the fundamental machine learning algorithms.  
The implementation uses **NumPy**, **Matplotlib**, and **scikit-learn** to train the model, predict values, and visualize results.

---

## 🎯 Objective

- Implement Linear Regression using `scikit-learn`
- Train a model using sample data
- Predict output values for known and new inputs
- Visualize the regression line with data points

---

## 🧠 Algorithm Overview

Linear Regression models the relationship between an input variable `X` and an output variable `y` using the equation:
y = mX + c


Where:
- `m` is the slope (coefficient)
- `c` is the intercept

---

## ⚙️ Implementation Details

1. Define input (`X`) and output (`y`) datasets
2. Create a `LinearRegression` model
3. Train the model using `.fit()`
4. Predict values using `.predict()`
5. Visualize results using Matplotlib

---

## 🧪 Sample Input

```python
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

✅ Output
Predicted value for input 6: 5.6
Slope (m): 0.6
Intercept (c): 2.2

## 📈 Visualization
Scatter Plot: Actual data points
Line Plot: Regression line predicted by the model
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/30c1912f-bfa1-4cf1-a0dd-dd1272b61518" />


## 🧩 Technologies Used
Python 3
NumPy
Matplotlib
scikit-learn



