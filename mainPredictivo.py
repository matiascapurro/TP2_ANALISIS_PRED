import pandas as pd
import json
import ast
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("origen.csv")
#df2 = pd.read_csv("testear.csv")
df.info()
df.describe()
df.head(50)
df.dtypes
dfPrueba = df.dropna()
dfPrueba.iloc[:, :15]
print(df.columns)

# Función para convertir valores JSON en una cadena de nombres separados por comas
def convertir_a_cadena(valor):
    try:
        # Si el valor es "nan" o una lista vacía, devolver una cadena vacía
        if valor == "nan" or valor == "[]" or pd.isna(valor):
            return ""
        
        # Intenta cargar el JSON y obtener los nombres de los génerosr)
        valor = valor.replace("'", "\"")
        json_data = json.loads(valor)
        if isinstance(json_data, list):
            nombres = [item["name"] for item in json_data]
            return ", ".join(nombres)
        else:
            return ""
    except (json.JSONDecodeError, TypeError):
        return ""

# Aplicar la función a la columna "genres_y" y crear una nueva columna "genres_str"
df['genres_str'] = df['genres_y'].apply(convertir_a_cadena)
df['companies_str'] = df['production_companies'].apply(convertir_a_cadena)
df['countries_str'] = df['production_countries'].apply(convertir_a_cadena)

#print(df['genres_str'].unique())
#print(df['companies_str'].unique())
print(df['countries_str'].unique())

longitud = len(df)
cantidad_nulos_por_columna = df.isnull().sum()
print("Porcentaje de valores nulos por columna:")
porcentaje_nulos = cantidad_nulos_por_columna/longitud
print(porcentaje_nulos)
print(df.dtypes)
df.head(20)

#BORRO LAS VARIABLES QUE NO ME INTERESAN
df = df.drop('genres_y',axis=1)
df = df.drop('production_companies',axis=1)
df = df.drop('production_countries',axis=1)
df = df.drop('adult',axis=1)
df = df.drop('budget',axis=1)
df = df.drop('original_language',axis=1)
df = df.drop('revenue',axis=1)
df = df.drop('runtime',axis=1)
df = df.drop('status',axis=1)
df = df.drop('tagline',axis=1)
df = df.drop('video',axis=1)
df = df.drop('genres_str',axis=1)
df = df.drop('companies_str',axis=1)
df = df.drop('countries_str',axis=1)
df = df.drop('popularity',axis=1)

df.to_csv('origen_limpio.csv', index=False)

df = pd.read_csv("origen_limpio.csv")

print(len(df))
longitud = 977541
cantidad_nulos_por_columna = df.isnull().sum()
print("Porcentaje de valores nulos por columna:")
porcentaje_nulos = cantidad_nulos_por_columna/longitud
print(porcentaje_nulos)
print(df.dtypes)
df.head(20)

#EDA
summary1 = df['averageRating'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(df['averageRating'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Average rating')
plt.ylabel('Frecuencia')
plt.title('Distribution')

plt.subplot(2, 2, 2)
plt.boxplot(df['averageRating'], vert=False)
plt.xlabel('average rating')
plt.title('Diagrama de Caja de averageRating')
print(summary1)

summary1 = df['numVotes'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.hist(df['numVotes'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Número de Votos')
plt.ylabel('Frecuencia')
plt.title('Distribución de numVotes')

plt.subplot(2, 2, 2)
plt.boxplot(df['numVotes'], vert=False)
plt.xlabel('Número de Votos')
plt.title('Diagrama de Caja de numVotes')
print(summary1)

todas_las_filas = len(df)

summary2 = df['startYear'].describe()

plt.subplot(2, 2, 3)
df['startYear'].plot(kind='hist', bins=30, color='lightcoral', edgecolor='black')
plt.xlabel('Año de Inicio')
plt.ylabel('Frecuencia')
plt.title('Distribución de startYear')

plt.subplot(2, 2, 4)
df['startYear'].plot(kind='box', vert=False)
plt.xlabel('Año de Inicio')
plt.title('Diagrama de Caja de startYear')
print(summary2)

año_min = 1900
año_max = 2023

df = df[(df['startYear'] >= año_min) & (df['startYear'] <= año_max)]
nuevas_filas = len(df)

diferencia = todas_las_filas - nuevas_filas
print(diferencia)

plt.subplot(4,4,2)
df['startYear'].plot(kind='hist', bins=30, color='lightcoral', edgecolor='black')
plt.xlabel('Año de Inicio')
plt.ylabel('Frecuencia')
plt.title('Distribución de startYear')

plt.subplot(4,4,4)
df['startYear'].plot(kind='box', vert=False)
plt.xlabel('Año de Inicio')
plt.title('Diagrama de Caja de startYear')

summary3 = df['endYear'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(4, 4, 2)
df['endYear'].plot(kind='hist', bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Año de Fin')
plt.ylabel('Frecuencia')
plt.title('Distribución de endYear')

plt.subplot(4, 4, 4)
df['endYear'].plot(kind='box', vert=False)
plt.xlabel('Año de Fin')
plt.title('Diagrama de Caja de endYear')
print(summary3)
len(df)
print(len(df[df['endYear'] == 0])/len(df))
#EL 97% DE LA COLUMMNA 'ENDYEAR' ES IGUAL A CERO, POR LO QUE LA DROPEO
df = df.drop('endYear', axis=1)
df.head(20)

summary4 = df['runtimeMinutes'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['runtimeMinutes'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Duración (minutos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de runtimeMinutes')

plt.subplot(2, 2, 2)
df['runtimeMinutes'].plot(kind='box', vert=False)
plt.xlabel('Duración (minutos)')
plt.title('Diagrama de Caja de runtimeMinutes')
print(summary4)

a = len(df)
df = df[df['runtimeMinutes'] >= 0]
df = df[df['runtimeMinutes'] <= 240]
df['runtimeMinutes'] = df['runtimeMinutes'].replace(0, np.nan)
b = len(df)
print((a-b)/a)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['runtimeMinutes'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Duración (minutos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de runtimeMinutes')

plt.subplot(2, 2, 2)
df['runtimeMinutes'].plot(kind='box', vert=False)
plt.xlabel('Duración (minutos)')
plt.title('Diagrama de Caja de runtimeMinutes')
print(df['runtimeMinutes'].describe())

summary5 = df['seasonNumber'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['seasonNumber'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Temporada')
plt.ylabel('Frecuencia')
plt.title('Distribución de seasonNumber')

plt.subplot(2, 2, 2)
df['seasonNumber'].plot(kind='box', vert=False)
plt.xlabel('Temporada')
plt.title('Diagrama de Caja de seasonNumber')
print(summary5)

print(len(df))
dfy = df[df['seasonNumber'] > 30]
dfNulos = df[df['seasonNumber'].isnull()]
dfMenores = df[df['seasonNumber'] < 30]
len(dfNulos)
len(dfMenores)
df = pd.concat([dfNulos, dfMenores], ignore_index=True)

summary5 = df['seasonNumber'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['seasonNumber'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Temporada')
plt.ylabel('Frecuencia')
plt.title('Distribución de seasonNumber')

plt.subplot(2, 2, 2)
df['seasonNumber'].plot(kind='box', vert=False)
plt.xlabel('Temporada')
plt.title('Diagrama de Caja de seasonNumber')
print(summary5)
print(len(df))

summary5 = df['seasonNumber'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['seasonNumber'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Temporada')
plt.ylabel('Frecuencia')
plt.title('Distribución de seasonNumber')

plt.subplot(2, 2, 2)
df['seasonNumber'].plot(kind='box', vert=False)
plt.xlabel('Temporada')
plt.title('Diagrama de Caja de seasonNumber')
print(summary5)

summary6 = df['episodeNumber'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['episodeNumber'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Episodio')
plt.ylabel('Frecuencia')
plt.title('Distribución de episodeNumber')

plt.subplot(2, 2, 2)
df['episodeNumber'].plot(kind='box', vert=False)
plt.xlabel('Episodio)')
plt.title('Diagrama de Caja de episodeNumber')
print(summary6)

print(len(df))
df9 = df[df['episodeNumber'].isnull()]
print(len(df9))
df['episodeNumber'] = df['episodeNumber'].apply(lambda x: np.nan if x > 50 else x)
print(len(df))
df9 = df[df['episodeNumber'].isnull()]
print(len(df9))

summary7 = df['ordering'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['ordering'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Ordenamiento')
plt.ylabel('Frecuencia')
plt.title('Distribución de ordering')

plt.subplot(2, 2, 2)
df['ordering'].plot(kind='box', vert=False)
plt.xlabel('Ordenamiento')
plt.title('Diagrama de Caja de ordering')
print(summary7)

print(len(df))
df9 = df[df['ordering'].isnull()]
print(len(df9))
df['ordering'] = df['ordering'].apply(lambda x: np.nan if x >= 20 else x)
print(len(df))
df9 = df[df['ordering'].isnull()]
print(len(df9))

summary8 = df['popularity'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['popularity'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Popularidad')
plt.ylabel('Frecuencia')
plt.title('Distribución de popularuty')

plt.subplot(2, 2, 2)
df['popularity'].plot(kind='box', vert=False)
plt.xlabel('Popularidad')
plt.title('Diagrama de Caja de popularity')
print(summary8)

print(len(df))
df9 = df[df['popularity'].isnull()]
print(len(df9))
df['popularity'] = df['popularity'].apply(lambda x: np.nan if x > 100 else x)
print(len(df))
df9 = df[df['popularity'].isnull()]
print(len(df9))

summary9 = df['isAdult'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['isAdult'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Es Adulto')
plt.ylabel('Frecuencia')
plt.title('Distribución de isAdult')

print(summary9)
df['isAdult'].unique()

summary10 = df['isOriginalTitle'].describe()
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
df['isOriginalTitle'].plot(kind='hist', bins=30, color='lightblue', edgecolor='black')
plt.xlabel('Es titulo original')
plt.ylabel('Frecuencia')
plt.title('Distribución de isOriginalTitle')

print(summary10)
df['isOriginalTitle'].unique()
#No sirve de nada esta
df = df.drop('isOriginalTitle', axis=1)

df.to_csv('origen_sin_outliers.csv', index=False)

df = pd.read_csv("origen_sin_outliers.csv")

print(len(df))
longitud = 977541
cantidad_nulos_por_columna = df.isnull().sum()
print("Porcentaje de valores nulos por columna:")
porcentaje_nulos = cantidad_nulos_por_columna/longitud
print(porcentaje_nulos)
print(df.dtypes)
df.head(20)

print(df['titleType'].unique())
value_counts = df['titleType'].value_counts()
plt.figure(figsize=(10, 6))
value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Histograma de Frecuencia de titleType')
plt.xlabel('titleType')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)  # Esto rota las etiquetas en el eje x para facilitar la lectura si hay muchos valores
plt.show()

print(df['genres_x'].unique())

# Dividir los géneros y crear una lista de todos los géneros
all_genres = []
for genres in df['genres_x']:
    if isinstance(genres, str):  # Solo procesar las cadenas
        genres_list = genres.split(',')
        all_genres.extend(genres_list)

# Contar la frecuencia de cada género
genre_counts = pd.Series(all_genres).value_counts()
print(genre_counts)

# Crear un histograma
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar')
plt.xlabel('Género')
plt.ylabel('Frecuencia')
plt.title('Histograma de Géneros')
plt.show()

# Reemplaza 0 con NaN en la columna 'genres_x'
df['genres_x'] = df['genres_x'].replace('0', np.nan)

all_directors = []
for directors in df['directors']:
    if isinstance(directors, str):  # Solo procesar las cadenas
        directors_list = directors.split(',')
        all_directors.extend(directors_list)

# Contar la frecuencia de cada género
df['directors'] = df['directors'].replace('0', np.nan)
directors_counts = pd.Series(all_directors).value_counts()
print(directors_counts)
directors_count = df['directors'].nunique()

print("Cantidad de directores únicos:", directors_count)

all_writers = []
for writers in df['writers']:
    if isinstance(writers, str):  # Solo procesar las cadenas
        writers_list = writers.split(',')
        all_writers.extend(writers_list)

# Contar la frecuencia de cada género
df['writers'] = df['writers'].replace('0', np.nan)
writers_counts = pd.Series(all_writers).value_counts()
print(writers_counts)
writers_count = df['writers'].nunique()

print("Cantidad de escritores únicos:", directors_count)

dfl = df[df['language'] == '0']
print("Cantidad de valores o nulos o cero: ")
print((len(dfl)/len(df))+0.61)
#EL 98% DE LOS LANGUAGES NO TIENEN NADA, DROPEO LA COLUMNA
df = df.drop('language', axis =1)

#0.61 nulos
dfa = df[df['attributes'] == '0']
print((len(dfa)/len(df))+0.61)
#96% DE LOS DATOS NULOS O EN CERO, DROPEO LA COLUMNA
df = df.drop('attributes', axis =1)
df = df.drop('Unnamed: 0', axis=1)

df.to_csv('origen_para_predecir.csv', index=False)

df = pd.read_csv('origen_para_predecir.csv')
df = df.drop('popularity', axis=1)
testeoKaggle = pd.read_csv('testear.csv')
testeoKaggle = testeoKaggle[['numVotes', 'titleType','isAdult', 'startYear', 'runtimeMinutes', 'genres_x', 'directors', 'writers', 'seasonNumber', 'episodeNumber', 'ordering']]
testeoKaggle['averageRating'] = 0

mean_runtimeMinutes = df['runtimeMinutes'].median()
mean_seasonNumber = df['seasonNumber'].median()
mean_episodeNumber = df['episodeNumber'].median()
mean_ordering = df['ordering'].median()


# Llena los valores nulos con las medias calculadas
df['runtimeMinutes'].fillna(mean_runtimeMinutes, inplace=True)
df['seasonNumber'].fillna(mean_seasonNumber, inplace=True)
df['episodeNumber'].fillna(mean_episodeNumber, inplace=True)
df['ordering'].fillna(mean_ordering, inplace=True)

testeoKaggle['runtimeMinutes'].fillna(mean_runtimeMinutes, inplace=True)
testeoKaggle['seasonNumber'].fillna(mean_seasonNumber, inplace=True)
testeoKaggle['episodeNumber'].fillna(mean_episodeNumber, inplace=True)
testeoKaggle['ordering'].fillna(mean_ordering, inplace=True)

train, test = train_test_split(df, test_size=0.1, random_state=42)

# Inicializa el codificador TargetEncoder
encoder = ce.TargetEncoder(cols=['titleType'])
# Ajusta el codificador a los datos de entrenamiento
encoder.fit(train, train['averageRating'])
# Aplica la codificación al conjunto de entrenamiento
train = encoder.transform(train)
# Aplica la codificación al conjunto de prueba
test = encoder.transform(test)
testeoKaggle = encoder.transform(testeoKaggle)

test['genre1'] = test['genres_x'].str.split(',').str[0]
train['genre1'] = train['genres_x'].str.split(',').str[0]
testeoKaggle['genre1'] = testeoKaggle['genres_x'].str.split(',').str[0]

encoder = ce.TargetEncoder(cols=['genre1'])
encoder.fit(train, train['averageRating'])

train = encoder.transform(train)
test = encoder.transform(test)
testeoKaggle = encoder.transform(testeoKaggle)

test['mainDirector'] = test['directors'].str.split(',').str[0]
train['mainDirector'] = train['directors'].str.split(',').str[0]
testeoKaggle['mainDirector'] = testeoKaggle['directors'].str.split(',').str[0]

encoder = ce.TargetEncoder(cols=['mainDirector'])
encoder.fit(train, train['averageRating'])

train = encoder.transform(train)
test = encoder.transform(test)
testeoKaggle = encoder.transform(testeoKaggle)

test['mainWriter'] = test['writers'].str.split(',').str[0]
train['mainWriter'] = train['writers'].str.split(',').str[0]
testeoKaggle['mainWriter'] = testeoKaggle['writers'].str.split(',').str[0]

encoder = ce.TargetEncoder(cols=['mainWriter'])
encoder.fit(train, train['averageRating'])

train = encoder.transform(train)
test = encoder.transform(test)
testeoKaggle = encoder.transform(testeoKaggle)

train = train.drop('genres_x', axis=1)
train = train.drop('directors', axis=1)
train = train.drop('writers', axis=1)

test = test.drop('genres_x', axis=1)
test = test.drop('directors', axis=1)
test = test.drop('writers', axis=1)

testeoKaggle = testeoKaggle.drop('genres_x', axis=1)
testeoKaggle = testeoKaggle.drop('directors', axis=1)
testeoKaggle = testeoKaggle.drop('writers', axis=1)
testeoKaggle = testeoKaggle.drop('averageRating', axis=1)

model = LinearRegression()
# Entrenar el modelo utilizando los datos de entrenamiento
X_train = train[['numVotes', 'titleType','isAdult', 'startYear', 'runtimeMinutes', 'seasonNumber', 'episodeNumber', 'ordering', 'genre1', 'mainDirector', 'mainWriter']]
y_train = train['averageRating']
model.fit(X_train, y_train)
# Realizar predicciones en los datos de prueba
X_test = test[['numVotes', 'titleType','isAdult', 'startYear', 'runtimeMinutes', 'seasonNumber', 'episodeNumber', 'ordering', 'genre1', 'mainDirector', 'mainWriter']]
y_test = test['averageRating']
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo en los datos de prueba
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")


# Crear un modelo de Random Forest
random_forest_model = RandomForestRegressor(n_estimators=600, random_state=42, max_depth=20)  # Puedes ajustar los hiperparámetros según tus necesidades

# Entrenar el modelo con tus datos de entrenamiento
random_forest_model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = random_forest_model.predict(X_test)  # Asumiendo que tienes un DataFrame de prueba X_test

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

new_predictions = random_forest_model.predict(testeoKaggle)
new_predictions = pd.DataFrame(new_predictions)
print(new_predictions.columns)
new_predictions.to_csv('submitionHoyKaggle.csv', index=True, header=['Id'])

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, 40, 50],
    # Agrega más hiperparámetros aquí
}
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mejores hiperparámetros:", best_params)
print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

data = train.copy()  # Suponiendo que 'train' es tu DataFrame de entrenamiento

# Calcula la matriz de correlación
correlation_matrix = data.corr()

# Configura el tamaño de la figura
plt.figure(figsize=(10, 8))

# Crea un mapa de calor de la matriz de correlación
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Configura la etiqueta de los ejes
plt.xlabel("Variables")
plt.ylabel("Variables")

# Muestra la figura
plt.show()

# Crear un objeto de modelo de Ridge
ridge_model = Ridge(alpha=1.0)  # Puedes ajustar el valor de alpha según tus necesidades

# Entrenar el modelo utilizando los datos de entrenamiento
ridge_model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred_ridge = ridge_model.predict(X_test)

# Evaluar el rendimiento del modelo
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge Regression:")
print(f"Error Cuadrático Medio (MSE): {mse_ridge}")
print(f"Coeficiente de Determinación (R^2): {r2_ridge}")

# Calcula el MSE en el conjunto de entrenamiento
y_pred_train = ridge_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)

# Calcula el MSE en el conjunto de prueba
y_pred_test = ridge_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"MSE en conjunto de entrenamiento: {mse_train}")
print(f"MSE en conjunto de prueba: {mse_test}")

#n_estimators, learning_rate y max_depth (ESTOS SON LOS HIPERPARAMETROS QUE HAY QUE IR MODIFICANDO)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=11)

# Entrenar el modelo en los datos de entrenamiento
xgb_model.fit(X_train, y_train)

# Realizar predicciones en los datos de prueba
y_pred = xgb_model.predict(X_test)

# Evaluar el modelo en los datos de prueba
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

hola = xgb_model.predict(testeoKaggle)
hola = pd.DataFrame(hola)
hola.to_csv('submition3Kaggle.csv', index=True, header=['Id'])

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Crear una instancia de tu modelo XGBoost con los hiperparámetros deseados
xgb_model = XGBRegressor(n_estimators=600, learning_rate=0.2, max_depth=11)

# Realizar validación cruzada con 5 particiones (puedes ajustar el número de particiones)
cross_val_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Calcular el error cuadrático medio promedio y convertirlo a positivo
mse_mean = -cross_val_scores.mean()

print(f"Error Cuadrático Medio promedio en validación cruzada: {mse_mean}")

# Definir un rango de hiperparámetros para buscar
param_grid = {
    'n_estimators': [600, 700, 800],
    'learning_rate': [0.2, 0.3, 0.4],
    'max_depth': [11,12,13]
}

xgb_model = XGBRegressor()

grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_

best_xgb_model = XGBRegressor(**best_params)

best_xgb_model.fit(X_train, y_train)

y_pred = best_xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mejores Hiperparámetros: {best_params}")
print(f"Error Cuadrático Medio (MSE): {mse}")
