# %%
#librerias
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import (
    train_test_split, GridSearchCV, TimeSeriesSplit
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, PowerTransformer, StandardScaler
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error 
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# %%
#Cargar datos y codificador para caracteres especiales (unicode_escape)
df = pd.read_csv("SeoulBikeData.csv", encoding='unicode_escape')

#Limpieza previa para simplificar nombres de variables
df.columns = [
    x.lower().\
        replace("(°c)", '').\
        replace("(%)", '').\
        replace(" (m/s)", '').\
        replace(" (10m)", '').\
        replace(" (mj/m2)", '').\
        replace("(mm)", '').\
        replace(" (cm)", '').\
        replace(" ", '_')
    for x in df.columns
    ]
    
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['month'] = df['date'].dt.month.astype(str)
df['weekend'] = df['date'].dt.weekday.apply(
    lambda x: 1 if x >= 5 else 0
).astype(str)

# %%
#Agregamos variable de retraso. Mejor RMSE previo es 374.11
#Retraso de una hora, tomando valor de rbc anterior y pasandolo abajo
df['count_lag_1'] = df ['rented_bike_count'].shift(1)

#Excepcion de la primera fila que no tiene valor anterior
df = df.dropna().reset_index(drop=True)

# %%
# Selección de variables y asignación de columnas
num_features = [
    'temperature',
    'humidity',
    'wind_speed',
    'visibility',
    'dew_point_temperature',
    'solar_radiation',
    'rainfall',
    'snowfall'
]
#Agrega variable de lag a num_features
num_features.append('count_lag_1')

cat_features = [
    'seasons',
    'holiday',
    'functioning_day',
    'hour',
    'month',
    'weekend'
]

# %%
#unificar las características numéricas y categóricas
X = df[num_features + cat_features]
y = df['rented_bike_count']

#División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42,
    shuffle=False
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

# %%
#Validación cruzada para KNN
tscv = TimeSeriesSplit(n_splits=5)

full_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', VarianceThreshold()),
        ('model', KNeighborsRegressor())
    ]
)

param_grid = [
    {
    'model': [KNeighborsRegressor()],
    'model__n_neighbors': [5, 10, 15, 20]

    },
    {
    'model': [Ridge()],
    'model__alpha': [0.1, 1.0, 10.0]
    },
    {
    'model': [RandomForestRegressor(random_state=42)],
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None]
    }
]

# %%
#Configuración de GridSearchCV
grid_search = GridSearchCV(
    
    estimator=full_pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

print("Iniciando GridSearchCV...")
grid_search.fit(X_train, y_train)

#Extracción del mejor modelo y evaluación en el conjunto de prueba
best_model = grid_search.best_estimator_
best_rmse = -grid_search.best_score_

print(f'Mejor modelo: {grid_search.best_params_["model"]}')
print(f'Mejor RMSE en validación cruzada: {best_rmse:.2f}')

# %%
#Generar graficas comparativas
def save_comparative_plot(model, X_data, y_real, title, filename):
    y_pred = model.predict(X_data)
    plt.plot(y_real.values[:150], label='Real', color='blue', alpha=0.6)
    plt.plot(y_pred[:150], label='Prediccion', color='orange', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# %%
save_comparative_plot(
    best_model, X_train, y_train, 
    'Comparativa: Real vs Modelo (Train Set)', 
    'comparative_actual_model_train_set.png'
)

save_comparative_plot(
    best_model, X_test, y_test, 
    'Comparativa: Real vs Modelo (Test Set)', 
    'comparative_actual_model_test_set.png'
)

print("Proceso Finalizado. Gráficas y modelo generados.")
# %%
#Archivo en formato pickle
with open('model_prediction_bikerpro.pk', 'wb') as f:
    pickle.dump(best_model, f)

print(
    f"\nMejor modelo guardado con RMSE: "
    f"{best_model.named_steps['model']}"
    f"y RMSE: {best_rmse:.2f}"
)

# %% 
#Probando el archivo .pk (Si funciono)
try:
    with open('model_prediction_bikerpro.pk', 'rb') as f:
        loaded_model = pickle.load(f)
    print("Modelo cargado exitosamente desde el archivo .pk")

    # Realizar una predicción de prueba
    model_type = type(loaded_model.named_steps['model']).__name__
    print(f"Tipo de modelo en el modelo cargado: {model_type}")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
