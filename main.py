import numpy as np
import pandas as pd

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
df.to_csv("ratings.csv", index=False)

print("✅ Dataset generado y guardado en ratings.csv")
print(df.head())
