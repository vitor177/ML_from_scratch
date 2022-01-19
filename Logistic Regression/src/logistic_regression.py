import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Logistic_Regression():
    def __init__(self, learning_rate, no_of_iterations):    
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) )) 

        dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m)*np.sum(Y_hat - self.Y)


        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, X):
        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) )) 
        Y_pred = np.where( Y_pred > 0.5, 1, 0)
        return Y_pred
        #return 1/(1+math.pow(math.e,-(X.dot(self.w)+self.b)))


df = pd.read_csv('../data/diabetes.csv')

features = df.drop(columns='Outcome', axis=1)

scaler = StandardScaler()
scaler.fit(features)
standardizes_data = scaler.transform(features)

features = standardizes_data
target = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.33, random_state=2)

model = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
#print(training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
#print(test_data_accuracy)

input_data = (5,166,72,19,175,25.8,0.587,52)

input_data = np.asarray(input_data)

input_data_reshaped = input_data.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)



prediction = model.predict(std_data)
print("Changes!!!!!")

print("A predição é: ", prediction)