import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#C:\Users\vitor\Documents\Siddhardhan\Linear Regression\data\salary_data.csv

class Linear_Regression():

    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Gradient Descent
        for i in range(self.no_of_iterations):
            self.update_weights()
            #print("Valor de w: ", self.w)
    
    def update_weights(self,):
        Y_prediction = self.predict(self.X)
        dw = - (2* (self.X.T).dot(self.Y-Y_prediction))/self.m
        db = - 2 * np.sum(self.Y-Y_prediction)/self.m


        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, X):
        return X.dot(self.w)+self.b


df = pd.read_csv('../data/salary_data.csv')

X = df.iloc[:,:-1].values
Y = df.iloc[:,1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=2)

model = Linear_Regression(learning_rate=0.02, no_of_iterations=10)
model.fit(X_train, Y_train)

print(model.w[0])
print(model.b)
print("Ola mundo!")

test_data_prediction = model.predict(X_test)
print(test_data_prediction)

#plt.scatter(X_test, Y_test, color='red')
#plt.plot(X_test, test_data_prediction, color='blue')
#plt.xlabel('Work Experiency')
#plt.ylabel('Salary')
#plt.title('Salary x Experience')
#plt.show()