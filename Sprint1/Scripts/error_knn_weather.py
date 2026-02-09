
#%%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# %%
#Data Load y codificador para caracteres especiales
df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')


#%%
#Selección de variables y asignación de columnas
X = df[['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 
        'Visibility (10m)']]
y = df['Rented Bike Count']

#%%
#División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )

#%%
#Definir valores de k-vecinos a evaluar
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]
rmse_values = []

#Bucle para entrenar y evaluar el modelo KNN con diferentes valores de k
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)
    print(f'k={k}, RMSE={rmse:.2f}')

#%%
#Visualización de RMSE para diferentes valores de k
min_rmse = min(rmse_values)
best_k = k_values[rmse_values.index(min_rmse)]

plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_values, marker='o', linestyle='--', color='b')
plt.axhline(y=min_rmse, color='r', linestyle='--', 
    label=f'Min RMSE: {min_rmse:.2f} at k={best_k}')

plt.title('RMSE vs. k Numero de vecinos')
plt.xlabel('k (Numero de vecinos)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

#Guardar la figura
plt.savefig('error_knn_weather.png')
print(f'Figura guardada como error_knn_weather.png')
print(f'Error minimo RMSE: {min_rmse:.2f} obtenido con k={best_k}')
plt.show()
