# Importación de bibliotecas necesarias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Carga de datos
df = pd.read_csv('tvmarketing.csv')

# Separación de las columnas de características (TV) y etiquetas (Sales)
df_x = df["TV"]
df_y = df["Sales"]

# Visualización de un gráfico de dispersión de TV contra Sales
plt.scatter(df_x, df_y);

# División de los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, train_size=0.7, random_state=0000)

# Reformateo de los datos para que tengan una dimensión adicional
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]

# Creación de un modelo de regresión lineal
model = LinearRegression()

# Entrenamiento del modelo utilizando los datos de entrenamiento
model.fit(X_train, y_train)

# Coeficiente de regresión (pendiente) y término de intercepción
model.coef_ 
model.intercept_

# Realización de predicciones utilizando el modelo entrenado en los datos de prueba
y_pred = model.predict(X_test)

# Gráfico de dispersión de TV vs. Sales
plt.scatter(df_x, df_y)

# Gráfico de la línea de regresión (predicciones) en rojo
plt.plot(X_test, y_pred, color="red", linewidth=2, linestyle="-")

# Generación de índices para visualización
c = [i for i in range(1, 61, 1)]

# Creación de una nueva figura de gráfico
fig = plt.figure()

# Gráfico de las ventas reales (purple) y las predicciones (yellow) en función del índice
plt.plot(c, y_test, color="purple", linewidth=2, linestyle="-")
plt.plot(c, y_pred, color="yellow", linewidth=2, linestyle="-")

# Título del gráfico
fig.suptitle('Actual and Predicted') 

# Etiquetas de ejes
plt.xlabel('Index')
plt.ylabel('Sales')

# Gráfico de los términos de error entre las ventas reales y las predicciones
c = [i for i in range(1, 61, 1)]
fig = plt.figure()
plt.plot(c, y_test - y_pred, color="red", linewidth=2, linestyle="-")

# Título del gráfico de errores
fig.suptitle('Error Terms') 

# Etiquetas de ejes para el gráfico de errores
plt.xlabel('Index')
plt.ylabel('ytest-ypred')

# Cálculo del Error Cuadrático Medio (MSE) y el coeficiente de determinación (R^2)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Impresión de las métricas de evaluación del modelo
print('MSE:', mse)
print('r squared:', r_squared)

# Gráfico de dispersión entre las ventas reales y las predicciones
plt.scatter(y_test, y_pred, c='blue')

# Etiquetas de ejes para el gráfico de dispersión
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# Mostrar una cuadrícula en el gráfico
plt.grid()
