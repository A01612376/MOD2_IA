import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

#DEFINIR UNA CLASE PARA LA REGRESIÓN LINEAL
class RegresionLineal:
    def __init__(self, a=1e-3, epochs=1000):  
        self.lr = a  # Learning rate
        self.epochs = epochs
        self.pesos = None 
        self.bias = None 

    def init_params(self):
        self.pesos = np.zeros(self.n_features)  #Inicialización de los pesos como un arreglo de ceros
        self.bias = 0 

    def update_params(self, dp, db):
        self.pesos -= self.lr * dp  #Actualización de los pesos
        self.bias -= self.lr * db  #Actualización del sesgo

    def get_prediction(self, X):
        return np.dot(X, self.pesos) + self.bias

    def get_gradients(self, X, y, y_pred):
        error = y_pred - y  #Calcular el error entre las predicciones y los valores reales
        dp = (1 / self.n_samples) * np.dot(X.T, error)  #Calcular el gradiente de los pesos
        db = (1 / self.n_samples) * np.sum(error)  #Calcular el gradiente del sesgo
        return dp, db

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape  #Obtener el número de muestras y características
        self.init_params() 

        #GRADIENTE DESCENDENTE
        for _ in range(self.epochs):
            y_pred = self.get_prediction(X)  
            dp, db = self.get_gradients(X, y, y_pred)  
            self.update_params(dp, db) 

    def predict(self, X):
        y_pred = self.get_prediction(X)  #Obtener predicciones
        return y_pred

#FUNCIÓN DE ERROR ROOT MEAN SQUARED ERROR
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))

#GENERAR DATOS SINTÉTICOS PARA REGRESIÓN
X, y = datasets.make_regression(
    n_samples=1000, n_features=1, noise=20, random_state=1
)

#DIVISIÓN DE LOS CONJUNTOS DE DATOS EN ENTRENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#INSTANCIA DEL MODELO DE REGRESIÓN LINEAL
linreg = RegresionLineal(a=0.01, epochs=1000)

#ENTRENAR EL MODELO EN EL CONJUNTO DE ENTRENAMIENTO
linreg.fit(X_train, y_train)

#PREDICCIONES EN EL CONJUNTO DE PRUEBA
predictions = linreg.predict(X_test)

print(f"RMSE: {rmse(y_test, predictions)}")

#GRÁFICA DE DISPERSIÓN PARA COMPARAR LOS VALORES REALES Y LAS PREDICCIONES
plt.scatter(y_test, predictions)
plt.show()
