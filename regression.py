import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv("heart.csv")


#check distribution
#print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
model = LogisticRegression()

model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accurecy = accuracy_score(X_train_prediction, Y_train)

print(training_data_accurecy)

X_test_prediction = model.predict(X_test)
test_data_accurecy = accuracy_score(X_test_prediction, Y_test)

print(test_data_accurecy)

input_data = (52,1,2,172,199,1,1,162,0,0.5,2,0,3)

input_as_np = np.asarray(input_data)

reshaped = input_as_np.reshape(1,-1)

prediction = model.predict(reshaped)
if (prediction[0] == 0):
    print('Person does not have heart disease')
else:
    print('Person has heart disease')

