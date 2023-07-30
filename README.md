# Logistic Regression with Regularization

This repository contains a Python script implementing logistic regression with regularization. Logistic regression is a popular binary classification algorithm widely used in machine learning. Regularization is introduced to prevent overfitting and improve the model's generalization performance.

## Dataset

The code utilizes two datasets:

1. **ex2data1.txt**
   - This dataset contains input data with two variables: Exam 1 scores and Exam 2 scores.
   - The output variable takes binary values, indicating whether a student was admitted or not admitted.

2. **ex2data2.txt**
   - This dataset contains input data with two variables: Microchip Test 1 and Test 2 scores.
   - The output variable takes binary values, indicating whether a microchip was accepted or rejected.

## Getting Started

To use this code, you need Python 3.x and the following dependencies:

- numpy
- matplotlib

You can install the required dependencies using pip:

```
pip install numpy matplotlib
```

Once the dependencies are installed, you can run the `C1_W3_Logistic_Regression.py` script to train and visualize the logistic regression model with regularization.

## Implementation Highlights

The code is divided into several parts:

1. **Data Loading and Visualization:** The script loads the training data from the provided files and visualizes it using matplotlib.

2. **Sigmoid Function:** The `sigmoid` function is implemented to compute the sigmoid activation used in logistic regression.

3. **Cost Function:** The `compute_cost` function calculates the cost of the logistic regression model using the negative log-likelihood loss.

4. **Gradient Calculation:** The `compute_gradient` function computes the gradient of the cost function with respect to the model's parameters (weights and bias).

5. **Gradient Descent:** The `gradient_descent` function performs gradient descent to optimize the model's parameters.

6. **Prediction:** The `predict` function is implemented to make predictions using the trained model.

7. **Feature Mapping:** Feature mapping is performed to handle non-linear decision boundaries in the `map_feature` function.

8. **Regularized Cost Function:** The `compute_cost_reg` function calculates the regularized cost of the logistic regression model.

9. **Regularized Gradient Calculation:** The `compute_gradient_reg` function computes the gradient of the regularized cost function.

## Training and Visualization

To train and visualize the logistic regression model, follow these steps:

1. Ensure the required dependencies are installed.

2. Run the `C1_W3_Logistic_Regression.py` script.

3. The script will load the data, perform feature mapping (for `ex2data2.txt`), and train the model using gradient descent with regularization.

4. After training, the script will visualize the decision boundary of the trained model.

## Results

The accuracy of the trained model on the training data will be displayed, providing an evaluation of the model's performance.
