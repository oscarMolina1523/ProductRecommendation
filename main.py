import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import networkx as nx


fake = Faker()      #generador de nombres falsos
np.random.seed(42)  #semilla

# Parámetros del dataset
n_users = 300       #cantidad de usuarios
n_products = 500    #cantidad de productos en nuestro caso peliculas
n_ratings = 6500    #cantidad de ratings que queremos

# Generar NOMBRES DE USUARIOS
user_names = {i: fake.name() for i in range(1, n_users+1)}

# Generar NOMBRES DE PRODUCTOS (nuestras peliculas)
product_names = {i: fake.sentence(nb_words=3).replace(".", "") 
                 for i in range(1, n_products+1)}

# Generamos datos aleatorios para los ids
user_ids = np.random.randint(1, n_users+1, n_ratings)
product_ids = np.random.randint(1, n_products+1, n_ratings)
ratings = np.random.randint(1, 6, n_ratings)
timestamps = np.random.randint(1609459200, 1640995200, n_ratings)

# Crear DataFrame con usernames incluidos
df = pd.DataFrame({
    "UserID": user_ids,
    "UserName": [user_names[uid] for uid in user_ids],
    "ProductID": product_ids,
    "ProductName": [product_names[iid] for iid in product_ids],
    "Rating": ratings,
    "Timestamp": timestamps
})

#mandamos a crear en un csv la data procesada
df.to_csv("data/ratings.csv", index=False)

print("Dataset generado con nombres reales:")
print(df.head())


#leemos los datos del csv procesado
df = pd.read_csv("data/ratings.csv")

#definimos un rango de ratings, nosotros usamos del 1 al 5 
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['UserID', 'ProductID', 'Rating']], reader)

#separamos entre entrenamiento y pruebas
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

#CREAMOS UN MODELO DE RECOMENDACIÓN SVD 
algo = SVD()

#ponemos al modelo a aprender con el 80% de los datos
#es decir el modelo actualiza gradualmente sus parámetros internos para reducir el error.
#Lo hace miles de veces hasta que el error es lo más pequeño posible.
algo.fit(trainset)

#una vez entrenado ahora lo probamos con el 20% de datos que no conoce
predictions = algo.test(testset)
#RMSE : error cuadratico medio entre mas bajo mejor
print("RMSE:", accuracy.rmse(predictions))

# Crear mapeos para usar en lo graficos y asi no mostrar los ids
user_id_to_name = df.set_index("UserID")["UserName"].to_dict()
product_id_to_name = df.set_index("ProductID")["ProductName"].to_dict()


#GENERAR TOP-N (3) RECOMENDACIONES 
def get_top_n(predictions, n=3):
    top_n = defaultdict(list) #creamos un diccionario cada usuario tendra una lista de pelis
    # Iteramos sobre todas las predicciones
    # Cada predicción tiene: uid=usuario, iid=producto, true_r=rating real, est=rating estimado, _=detalles internos
    for uid, iid, true_r, est, _ in predictions:
        # Agregamos a la lista del usuario la tupla (ProductoID, Rating Estimado)
        top_n[int(uid)].append((int(iid), est))

    # Para cada usuario, ordenamos los productos por rating estimado de mayor a menor
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # Nos quedamos solo con los n mejores productos (top-N)
        top_n[uid] = user_ratings[:n]

    # Devolvemos el diccionario con los top-N por usuario
    return top_n

all_predictions = []                   # Lista vacía para guardar todas las predicciones
all_user_ids = df['UserID'].unique()   # Obtenemos todos los IDs de usuario únicos
all_item_ids = df['ProductID'].unique()# Obtenemos todos los IDs de productos únicos

# Recorremos cada usuario
for uid in all_user_ids:
    # Obtenemos los productos que el usuario ya calificó
    items_rated = df[df['UserID']==uid]['ProductID'].values
    # Seleccionamos solo los productos que el usuario NO ha calificado
    items_to_predict = [iid for iid in all_item_ids if iid not in items_rated]
    
    # Generamos predicciones para cada producto que aún no ha visto el usuario
    for iid in items_to_predict:
        # Se usa el modelo entrenado 'algo' para predecir el rating estimado
        all_predictions.append(algo.predict(uid, iid))


top_n = get_top_n(all_predictions, n=3)  

# G R Á F I C O   1 
# Distribución de ratings 

plt.figure(figsize=(8, 5))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribución de Ratings Asignados (Dataset Inicial)', fontsize=16)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Cantidad de Ratings', fontsize=12)
plt.show()

# G R Á F I C O   2 
# Frecuencia de productos recomendados

recommended_products = []
for uid, user_ratings in top_n.items():
    for iid, _ in user_ratings:
        recommended_products.append(iid)

rec_df = pd.Series(recommended_products).value_counts().reset_index()
rec_df.columns = ["ProductID", "Frecuencia"]
rec_df["ProductName"] = rec_df["ProductID"].map(product_id_to_name)

rec_df_top = rec_df.head(20)

plt.figure(figsize=(14, 7))
sns.barplot(
    x="ProductName",
    y="Frecuencia",
    data=rec_df_top,
    palette="magma"
)
plt.title("Frecuencia de Películas en el Top-3", fontsize=18)
plt.xlabel("Película")
plt.ylabel("Frecuencia")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# G R Á F I C O   3
# Matriz de ratings estimados (heatmap)

sample_uids = list(top_n.keys())[:5]
sample_predictions_data = {}

for uid in sample_uids:
    sample_predictions_data[user_id_to_name[uid]] = {
        product_id_to_name[iid]: est for iid, est in top_n[uid]
    }

pred_df = pd.DataFrame.from_dict(sample_predictions_data, orient="index").fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(pred_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title("Ratings Estimados del TOP-3 ", fontsize=16)
plt.xlabel("Película")
plt.ylabel("Usuario")

plt.xticks(rotation=45)   # gira nombres de películas
plt.yticks(rotation=15)   # gira un poco los usuarios

plt.tight_layout()

plt.show()

#G R Á F I C O   4 
# Barras: Top-3 por usuario

# sample_users = list(top_n.keys())[:5]

# sample_recommendations = []
# for uid in sample_users:
#     for iid, est in top_n[uid]:
#         sample_recommendations.append({
#             "UserName": user_id_to_name[uid],
#             "ProductName": product_id_to_name[iid],
#             "EstimatedRating": est
#         })

# sample_df = pd.DataFrame(sample_recommendations)

# plt.figure(figsize=(12, 6))
# sns.barplot(
#     x="UserName",
#     y="EstimatedRating",
#     hue="ProductName",
#     data=sample_df,
#     palette="Set2"
# )
# plt.title("Top-3 Películas Recomendadas por Usuario", fontsize=16)
# plt.xlabel("Usuario")
# plt.ylabel("Rating Estimado")
# plt.xticks(rotation=45, ha="right")
# plt.legend(title="Película", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
# plt.show()

#G R Á F I C O   5 
# Heatmap grande con 50 usuarios × 20 películas

sample_users = list(top_n.keys())[:50]
top_products = rec_df["ProductID"].head(20).tolist()

heatmap_data = []
for uid in sample_users:
    row = []
    user_dict = dict(top_n[uid])
    for pid in top_products:
        row.append(user_dict.get(pid, 0))
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(
    heatmap_data,
    index=[user_id_to_name[u] for u in sample_users],
    columns=[product_id_to_name[p] for p in top_products]
)

plt.figure(figsize=(16, 10))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='black'
)
plt.title("Heatmap: Top-3 Películas Recomendadas", fontsize=16)
plt.xlabel("Película")
plt.ylabel("Usuario")
plt.tight_layout()
plt.show()


# 2) Top 20 Productos más Recomendados
# product_counts = pd.Series([iid for _, ur in top_n.items() for iid, _ in ur]).value_counts().head(20)
# product_counts.index = product_counts.index.map(product_id_to_name)  # mapear a nombres

# plt.figure(figsize=(14,6))
# ax = sns.barplot(x=product_counts.index, y=product_counts.values, palette="magma")
# plt.title("Top 20 Películas más Recomendadas (frecuencia)", fontsize=16)
# plt.xlabel("Película", fontsize=12)
# plt.ylabel("Número de Usuarios a los que se les recomendó", fontsize=12)
# plt.xticks(rotation=45, ha='right')

# # Añadir etiquetas de valor encima de cada barra
# for p in ax.patches:
#     height = p.get_height()
#     ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
#                 ha='center', va='bottom', fontsize=10)

# plt.tight_layout()
# plt.show()

#GRAFICO 6
# Red de Recomendaciones Usuario–Producto 
# Para que la red sea legible tomamos una muestra:
max_users_in_graph = 40   # cantidad de usuarios que quiero
sample_users = list(top_n.keys())[:max_users_in_graph]

edges = []
for uid in sample_users:
    for iid, est in top_n[uid]:
        # nodos con nombres 
        user_name = user_id_to_name.get(uid, f"U{uid}")
        prod_name = product_id_to_name.get(iid, f"P{iid}")
        edges.append((user_name, prod_name))

G = nx.Graph()
G.add_edges_from(edges)

plt.figure(figsize=(14, 10))
plt.title(f"Red de Recomendaciones (muestra {len(sample_users)} usuarios)", fontsize=16)

# Ponemos diferente color/forma a usuarios y productos
user_nodes = [n for n in G.nodes if n in {user_id_to_name.get(u) for u in sample_users}]
product_nodes = [n for n in G.nodes if n not in user_nodes]

pos = nx.spring_layout(G, k=0.5, seed=42)  # layout más compacto
nx.draw_networkx_edges(G, pos, alpha=0.15)

nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color="skyblue", node_size=120, label="Usuarios")
nx.draw_networkx_nodes(G, pos, nodelist=product_nodes, node_color="salmon", node_size=120, label="Productos")

# Dibujar etiquetas solo para los nodos más importantes 
# elegir top productos más recomendados y los usuarios muestreados
top_product_names = rec_df["ProductID"].head(15).map(product_id_to_name).tolist()
label_nodes = set(top_product_names) | set(list(user_nodes)[:15])  # etiquetas selectivas

labels = {n: n for n in G.nodes if n in label_nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

plt.legend(scatterpoints=1, loc="upper right")
plt.axis("off")
plt.tight_layout()
plt.show()