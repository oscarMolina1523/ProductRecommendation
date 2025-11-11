import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Fijamos una semilla para reproducibilidad y asi nos genere siempre los mismos datos
np.random.seed(42)

# Parámetros para nuestro dataset
n_users = 300       # número de usuarios
n_products = 500    # número de productos
n_ratings = 6500    # cantidad de ratings

# Generamos datos aleatorios
user_ids = np.random.randint(1, n_users+1, n_ratings)
product_ids = np.random.randint(1, n_products+1, n_ratings)
ratings = np.random.randint(1, 6, n_ratings)  # ratings entre 1 y 5
timestamps = np.random.randint(1609459200, 1640995200, n_ratings)  
# (fechas entre 2021-01-01 y 2022-01-01) para simular que son viejos

# Creamos DataFrame
df = pd.DataFrame({
    "UserID": user_ids,
    "ProductID": product_ids,
    "Rating": ratings,
    "Timestamp": timestamps
})

# Guardamos en CSV
df.to_csv("data/ratings.csv", index=False)

print("Dataset generado y guardado en ratings.csv")
print(df.head())

# Leemos el dataset
df = pd.read_csv("data/ratings.csv")

# Definir rango de ratings (entre 1 y 5 vamos a usar)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserID', 'ProductID', 'Rating']], reader)

# Separar entre entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Modelo SVD
algo = SVD()
algo.fit(trainset)

# Evaluamos
predictions = algo.test(testset)
print("RMSE:", accuracy.rmse(predictions))

#generamos un top de recomendaciones por usuario
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

## Gráfico 1: Distribución de Ratings Originales
# Muestra cómo están distribuidos los ratings en el dataset inicial.

plt.figure(figsize=(8, 5))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribución de Ratings Asignados (Dataset Inicial)', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Cantidad de Ratings', fontsize=12)
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['1', '2', '3', '4', '5'])
plt.show()



## Gráfico 2: Frecuencia de Productos Recomendados (Top-3)
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
rec_df_top = rec_df.head(20) 
plt.figure(figsize=(14, 7))
sns.barplot(
    x='ProductID',
    y='Frecuencia',
    data=rec_df_top,
    palette='magma', 
    order=rec_df_top['ProductID']
)

plt.title('Frecuencia de Aparición de Productos en el TOP-3 de Recomendaciones', fontsize=18, pad=20)
plt.xlabel('ID del Producto', fontsize=14)
plt.ylabel('Número de Veces Recomendado', fontsize=14)

plt.xticks(
    rotation=45, # Rota las etiquetas 45 grados
    ha='right'   # Alinea las etiquetas a la derecha para que no se choquen
)
plt.gca().set_xticklabels(rec_df_top['ProductID'].astype(int))
# Agregar etiquetas de frecuencia sobre las barras
for index, row in rec_df_top.iterrows():
    plt.text(rec_df_top.index.get_loc(index), row['Frecuencia'] + 0.1, row['Frecuencia'], color='black', ha="center", fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


## Gráfico 3: Matriz de Ratings Estimados
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


## Gráfico 4: Recomendaciones de una muestra de 5 usuarios
# Visualiza qué productos fueron recomendados a 5 usuarios seleccionados

# 1. Seleccionar una muestra de 5 usuarios
sample_users = list(top_n.keys())[:5]

# 2. Crear una lista con las recomendaciones de cada usuario
sample_recommendations = []
for uid in sample_users:
    for iid, est in top_n[uid]:
        sample_recommendations.append({"UserID": uid, "ProductID": iid, "EstimatedRating": est})

sample_df = pd.DataFrame(sample_recommendations)

# 3. Graficar
plt.figure(figsize=(10, 6))
sns.barplot(
    x="UserID",
    y="EstimatedRating",
    hue="ProductID",
    data=sample_df,
    palette="Set2"
)

plt.title("Top-3 de Productos Recomendados por Usuario (Muestra de 5)", fontsize=16)
plt.xlabel("ID de Usuario", fontsize=12)
plt.ylabel("Rating Estimado", fontsize=12)
plt.legend(title="ID Producto", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

## Gráfico 5: Red de Recomendaciones Usuario–Producto
# Visualiza todos los usuarios (300) conectados con los productos recomendados (Top-3 por usuario)

import networkx as nx

# 1. Construir los enlaces de la red
edges = []
for uid, user_ratings in top_n.items():
    for iid, est in user_ratings:
        edges.append((f"U{uid}", f"P{iid}"))  # prefijos para distinguir nodos

# 2. Crear el grafo bipartito
G = nx.Graph()
G.add_edges_from(edges)

# 3. Separar nodos por tipo (usuarios vs productos)
user_nodes = [n for n in G.nodes if n.startswith("U")]
product_nodes = [n for n in G.nodes if n.startswith("P")]

# 4. Asignar posiciones para visualización
pos = {}
pos.update((n, (1, i)) for i, n in enumerate(user_nodes))     # Usuarios en una columna
pos.update((n, (2, i)) for i, n in enumerate(product_nodes))  # Productos en otra

# 5. Crear la figura
plt.figure(figsize=(14, 16))
plt.title("Red de Recomendaciones Usuario–Producto (Top-3 por Usuario)", fontsize=18, pad=20)

# Dibujar nodos y aristas
nx.draw_networkx_edges(G, pos, alpha=0.2)

# Nodos de usuarios
nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color="skyblue", node_size=20, label="Usuarios")

# Nodos de productos
nx.draw_networkx_nodes(G, pos, nodelist=product_nodes, node_color="salmon", node_size=20, label="Productos")

# 6. Leyenda y estilo
plt.legend(scatterpoints=1, loc="upper right", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.show()

# Contar cuántos usuarios recibieron cada producto en top 3
product_counts = pd.Series([iid for _, ur in top_n.items() for iid, _ in ur]).value_counts().head(20)
plt.figure(figsize=(14,6))
sns.barplot(x=product_counts.index, y=product_counts.values, palette="magma")
plt.title("Top 20 Productos más Recomendados")
plt.xlabel("ID Producto")
plt.ylabel("Cantidad de Usuarios")
plt.xticks(rotation=45)
plt.show()

## Gráfico 4: Heatmap de Recomendaciones para múltiples usuarios
# Seleccionamos los 50 primeros usuarios para que sea legible
sample_users = list(top_n.keys())[:50]

# Identificar los productos más recomendados en todo el top_n
top_products = pd.Series([iid for _, ur in top_n.items() for iid, _ in ur]).value_counts().head(20).index

# Construir matriz: filas=usuarios, columnas=productos, valor=rating estimado
heatmap_data = []
for uid in sample_users:
    row = []
    user_dict = dict(top_n[uid])
    for pid in top_products:
        row.append(user_dict.get(pid, 0))  # 0 si no está recomendado
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=sample_users, columns=top_products)

plt.figure(figsize=(16, 10))
sns.heatmap(
    heatmap_df,
    annot=True,       # Mostrar valores dentro de celdas
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='black'
)
plt.title("Heatmap: Top-3 Productos Recomendados por Usuario", fontsize=16)
plt.xlabel("ID Producto", fontsize=12)
plt.ylabel("ID Usuario", fontsize=12)
plt.tight_layout()
plt.show()
