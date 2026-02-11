# %%
#librerias
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 

# %%
#Cargar datos y codificador para caracteres especiales (unicode_escape)
df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')
#Convertir la columna 'Date' a formato datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# %%
# Featture engineering
"""
Crear nuevas características basadas en la columna 'Date'
Extraer mes y día de la semana para capturar fin de semana. 
Convertir a string para one-hot encoding.

"""
df['Month'] = df['Date'].dt.month.astype(str)
df['Weekend'] = df['Date'].dt.weekday.apply(
    lambda x: 1 if x >= 5 else 0
).astype(str)

# %%
# Selección de variables y asignación de columnas
num_features = [
    'Temperature(°C)',
    'Humidity(%)',
    'Wind speed (m/s)', 
    'Visibility (10m)',
    'Dew point temperature(°C)', 
    'Solar Radiation (MJ/m2)',
    'Rainfall(mm)',
    'Snowfall (cm)' #Snowfall lleva espacio antes de cm
]

cat_features = [
    'Month',
    'Weekend',
    'Seasons',
    'Holiday',
    'Functioning Day',
    'Hour'
]

# %%
#unificar las características numéricas y categóricas
X = df[num_features + cat_features]
y = df['Rented Bike Count']

#División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
#Pipeline
#Preprocesamiento para características numéricas
#transformación Yeo-Johnson para hacer los datos mas normales
num_transformer = Pipeline(
    steps=[
        ('power', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ]
)

# %%
#Preprocesamiento para características categóricas
cat_transformer = Pipeline(
    steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
#Combinar preprocesamiento numérico y categórico
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

#Pipeline completo con preprocesamiento y modelo KNN
#Evaluacion con varios modelos k-vecinos
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]
best_rmse = float('inf')
best_model = None

for k in k_values:
    model_pipeline = Pipeline(
        steps=[
            ('preprocesssor', preprocessor),
            ('selector', VarianceThreshold()),
            ('regressor', KNeighborsRegressor(n_neighbors=k))
        ]
    )

# %%
    #entrenamiento del modelo
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f'Valor k evaluado={k}, Valor RMSE={rmse:.2f}')

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model_pipeline

# %%
# Guardar el mejor modelo en un archivo pickle
with open('model_fe_engineering_selection.pk', 'wb') as f:
    pickle.dump(best_model, f)

print(
    f"\nMejor modelo guardado con RMSE: "
    f"{best_model.named_steps['regressor'].n_neighbors}"
    f"y RMSE: {best_rmse:.2f}"
)

# %%
#Probando el archivo .pk (Si funciono)

try:
    with open('model_fe_engineering_selection.pk', 'rb') as f:
        loaded_model = pickle.load(f)
    print("Modelo cargado exitosamente desde el archivo .pk")

    # Realizar una predicción de prueba
    k_test = loaded_model.named_steps['regressor'].n_neighbors
    print(f"Valor de k en el modelo cargado: {k_test}")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# %%
