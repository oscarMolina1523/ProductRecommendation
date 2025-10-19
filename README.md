<h1 align="left">Product Recommendation System</h1>

###

<div align="center">
  <img height="350" src="https://i.ibb.co/2YS8DwS4/ratings-Puntajes.png"  />
</div>

###

<p align="left">Este proyecto implementa un <strong>sistema de recomendación de productos para una tienda online</strong>, utilizando <strong>filtrado colaborativo basado en el historial de usuarios</strong>. El objetivo es generar recomendaciones personalizadas <strong>(“top-N productos”)</strong> basadas en los ratings que los usuarios le han dado a productos.</p>

###

<h2 align="left">Contexto</h2>

###

<p align="left">-Dataset con columnas: "UserID", "ProductID", "Rating", "Timestamp".  <br>- La tienda desea sugerir al usuario productos que posiblemente le gusten, basándose en su comportamiento pasado y en el de otros usuarios.  <br>- Los métodos utilizados incluyen: factorización de matrices (SVD) y medidas de precisión como RMSE y ranking top-N.</p>

###

<h2 align="left">Metodología</h2>

###

<p align="left">1. Carga del dataset y preprocesamiento.  <br>2. División en conjunto de entrenamiento y prueba.  <br>3. Entrenamiento de un modelo SVD (descomposición matricial) para predecir ratings.  <br>4. Evaluación del modelo mediante RMSE en el conjunto de prueba.  <br>5. Generación de recomendaciones top-N para cada usuario, prediciendo productos no evaluados por él y ordenándolos por valor estimado.</p>

###

<h2 align="left">Instalación</h2>

###

<p align="left">## Instalación y uso  <br>1. Clona el repositorio:  <br>   ```bash<br>   git clone https://github.com/oscarMolina1523/ProductRecommendation.git<br>   cd ProductRecommendation<br>pip install -r requirements.txt<br>python main.py</p>

###

<h2 align="left">Resultados esperados</h2>

###

<p align="left">Impresión del valor de RMSE para la evaluación del conjunto de prueba.<br><br>Por cada usuario del dataset, una lista de productos recomendados (top-N) con sus estimaciones de rating.<br><br>CSV con el dataset simulado (si se generó) y posibles resultados adicionales que quieras guardar o exportar.</p>

###

<h2 align="left">Cómo interpretar los resultados</h2>

###

<p align="left">Un RMSE más bajo implica que el modelo predice ratings más cercanos a los reales.<br><br>Las recomendaciones “top-N” se obtienen seleccionando los productos que cada usuario no ha evaluado y ordenándolos según la estimación del modelo.<br><br>Puedes ajustar N (por ejemplo a 3, 5 o 10) para obtener más o menos recomendaciones por usuario.</p>

###

<h2 align="left">✨ Autores</h2>

###

<p align="left">Desarrollador Oscar Molina<br>💼 Desarrollador Web<br>GitHub: @oscarMolina1523<br>linkedin: https://www.linkedin.com/in/oscar-molina-916195309</p>

###
