import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict

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

for uid, user_ratings in top_n.items():
    print(f"Usuario {uid} → Recomendaciones: {[iid for (iid, _) in user_ratings]}")