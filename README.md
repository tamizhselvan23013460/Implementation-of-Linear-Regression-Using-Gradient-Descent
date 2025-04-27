# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset from '50_Startups.csv'.
2. Extract features 'x' and target 'y'.
3. Convert 'x' to float type and standardize using StandardScaler.
4. Standardize the target 'y' using StandardScaler.
5. Add a bias term (column of ones) to the feature matrix 'x1'.
6. Initialize the parameter vector 'theta' to zeros.
7. For each iteration in the specified range (num_iters): a. Compute predictions: y_hat = x.dot(theta). b. Compute the error: error = y_hat - y. c. Update theta using the gradient descent formula: theta = theta - (learning_rate / m) * x.T.dot(error).
8. Scale new input data using the fitted scaler.
9. Add bias term (1) to the scaled new data.
10. Compute the predicted value using the new scaled data and theta: prediction = np.dot(new_data, theta).
11. Inverse scale the prediction to get the final output.
12. Print the predicted value. .

## Program :
```
/*
Program to implement the linear regression using gradient descent.
Developed by: TAMIZHSELVAN B
RegisterNumber: 212223230225  
*/
```

```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X=np.c_[np.ones(len(X1)), X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors= (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
New_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
New_Scaled=scaler.fit_transform(New_data)
prediction=np.dot(np.append(1,New_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}")
```


## Output:

# Preview of datasets:

![EX_3_OUTPUT_1](https://github.com/user-attachments/assets/d7c60ba1-d1ca-4bd5-8c8f-20a4c8197c04)

# X Initialize : 
![EX_3_OUTPUT_2](https://github.com/user-attachments/assets/c3f0e889-d85f-4c3a-a5be-3dbb40fef4a0)

# Y Initialize : 
![EX_3_OUTPUT_3](https://github.com/user-attachments/assets/b0d56d73-fee0-43de-91b5-3cda26dfbfeb)

# X1 and Y1 Scaled Value :

![EX_3_OUTPUT_4](https://github.com/user-attachments/assets/0ff04bae-1e4f-441f-9831-dd0c0be00c91)

# Predicted Value:

![EX_3_OUTPUT_5](https://github.com/user-attachments/assets/a0db1dd1-c038-4412-a191-78942f67b1d9)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
