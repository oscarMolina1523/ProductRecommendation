import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Parámetros del dataset
n_users = 20       # número de usuarios
n_products = 15    # número de productos
n_ratings = 100    # cantidad de ratings

# Generar datos aleatorios
user_ids = np.random.randint(1, n_users+1, n_ratings)
product_ids = np.random.randint(1, n_products+1, n_ratings)
ratings = np.random.randint(1, 6, n_ratings)  # ratings entre 1 y 5
timestamps = np.random.randint(1609459200, 1640995200, n_ratings)  
# (fechas entre 2021-01-01 y 2022-01-01 en formato UNIX)

# Crear DataFrame
df = pd.DataFrame({
    "UserID": user_ids,
    "ProductID": product_ids,
    "Rating": ratings,
    "Timestamp": timestamps
})

# Guardar en CSV
df.to_csv("data/ratings.csv", index=False)

print("✅ Dataset generado y guardado en ratings.csv")
print(df.head())

# Leer el dataset
df = pd.read_csv("data/ratings.csv")

# Definir rango de ratings
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserID', 'ProductID', 'Rating']], reader)

# Separar train/test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Modelo SVD
algo = SVD()
algo.fit(trainset)

# Evaluar
predictions = algo.test(testset)
print("RMSE:", accuracy.rmse(predictions))


def get_top_n(predictions, n=5):
    '''Devuelve top-N recomendaciones por usuario'''
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Crear predicciones para todo el dataset
all_predictions = []
all_user_ids = df['UserID'].unique()
all_item_ids = df['ProductID'].unique()

for uid in all_user_ids:
    items_rated = df[df['UserID']==uid]['ProductID'].values
    items_to_predict = [iid for iid in all_item_ids if iid not in items_rated]
    for iid in items_to_predict:
        all_predictions.append(algo.predict(uid, iid))

top_n = get_top_n(all_predictions, n=3)
print("\n--- Top-3 Recomendaciones por Usuario ---")
for uid, user_ratings in top_n.items():
    print(f"Usuario {uid} → Recomendaciones: {[iid for (iid, _) in user_ratings]}")

## 📈 Gráfico 1: Distribución de Ratings Originales
# Muestra cómo están distribuidos los ratings en tu dataset inicial.

plt.figure(figsize=(8, 5))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribución de Ratings Asignados (Dataset Inicial)', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Cantidad de Ratings', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['1', '2', '3', '4', '5'])
plt.show()



## 📊 Gráfico 2: Frecuencia de Productos Recomendados (Top-3)
# Muestra qué productos son los más populares en las recomendaciones.

# 1. Extraer todos los IDs de productos recomendados
recommended_products = []
for uid, user_ratings in top_n.items():
    for iid, _ in user_ratings:
        recommended_products.append(iid)

# 2. Contar la frecuencia y preparar el DataFrame para el gráfico
rec_counts = pd.Series(recommended_products).value_counts()
rec_df = rec_counts.reset_index()
rec_df.columns = ['ProductID', 'Frecuencia']
rec_df = rec_df.sort_values(by='Frecuencia', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(
    x='ProductID',
    y='Frecuencia',
    data=rec_df,
    palette='magma', 
    order=rec_df['ProductID']
)

plt.title('Frecuencia de Aparición de Productos en el TOP-3 de Recomendaciones', fontsize=18, pad=20)
plt.xlabel('ID del Producto', fontsize=14)
plt.ylabel('Número de Veces Recomendado', fontsize=14)

# Agregar etiquetas de frecuencia sobre las barras
for index, row in rec_df.iterrows():
    plt.text(index, row['Frecuencia'] + 0.1, row['Frecuencia'], color='black', ha="center", fontsize=12)

plt.xticks(rec_df.index, rec_df['ProductID'].astype(int))
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# ---

## 🔥 Gráfico 3: Matriz de Ratings Estimados (Heatmap)
# Muestra los ratings estimados reales que usa el modelo para decidir el Top-3 (ejemplo de 5 usuarios).

# 1. Preparar datos para 5 usuarios de ejemplo
sample_uids = list(top_n.keys())[:5]
sample_predictions_data = {}
for uid in sample_uids:
    # Mapea ProductID a Rating Estimado (est)
    sample_predictions_data[uid] = {iid: est for iid, est in top_n[uid]}

# 2. Crear y limpiar el DataFrame
pred_df = pd.DataFrame.from_dict(sample_predictions_data, orient='index').fillna(0)
pred_df.index.name = "UserID"
pred_df = pred_df.sort_index(axis=1) # Ordenar por ID de producto

plt.figure(figsize=(10, 8))
sns.heatmap(
    pred_df,
    annot=True,        # Muestra el valor del rating estimado
    cmap="YlGnBu",     # Escala de color
    fmt=".2f",         # Formato de dos decimales
    linewidths=.5,
    linecolor='black'
)
plt.title('Ratings Estimados del TOP-3 Recomendado para 5 Usuarios', fontsize=16)
plt.ylabel('ID de Usuario', fontsize=12)
plt.xlabel('ID de Producto Recomendado', fontsize=12)
plt.show()
