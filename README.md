# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Startv the program.
2.import numpy as np.
3.Give the header to the data.
4.Find the profit of population.
5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6.End the program.
```
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

![EX_3_OUTPUT_5](https://github.com/user-attachments/assets/998fb07e-02e4-40fa-9415-6193b990bc8f)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
