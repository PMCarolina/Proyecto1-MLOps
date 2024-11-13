# Proyecto de Machine Learning: "Modelo de recomendacion de peliculas"

### Descripcion:
- En el contexto de un ambiente de trabajo en una start-up que provee servicios de agregación de plataformas de streaming, como rol de Data Scientist nuestro objetivo es crear un modelo de ML sobre un sistema de recomendación de peliculas que aún no ha sido implementado.

### Propuesta de trabajo:
- Con la informacion proveniente del dataset que nos proporciona la empresa nuestro trabajo comienza con la limpieza de datos, transformaciones especificas de los mismos, para luego realizar el desarrollo de una API con las consultas requeridas. Ya con los datos limpios se realiza un EDA (Analisis Exploratorio de los Datos), y finalmente con toda la data consumible, se entrena el modelo de machine learning para armar nuestro modelo de recomendacion de peliculas.


### Data
 * Los datos originales no se encuentran en este repositorio debido a su gran tamaño pero estan disponibles en:
https://drive.google.com/drive/folders/1X_LdCoGTHJDbD28_dJTxaD4fVuQC9Wt5

### Requisitos

* python
* sklearn
* TfidfVectorizer
* pandas
* numpy
* matplotlib
* seaborn
* Render
* FastApi

### Transfromaciones
   Movies dataset

* Se rellenaron los valores nulos con 0 en las columnas revenue y budget.

* Se eliminaron los campos nulos de release_date.

* Se cambió el tipo de dato a fecha (AAAA-mm-dd) en la columna release_year.

* Se creó una nueva columna llamada return que proviene de la división de revenue y budget.

* Se eliminaron las columnas innecesarias: video, imdb_id, adult, original_title, poster_path, homepage, overview, status, tagline y Unnamed: 0.

* También se desanidaron las columnas genres, production_companies, spoken_languages_iso y belongs_to_collection, extrayendo solo la información de interés de cada una.
  

 Credits dataset

* Se extrajeron las columnas anidadas dentro de las columnas cast y crew.

* Se realizó un chequeo de filas duplicadas.


## Funciones de la API

En la API creada se le incluyeron siete funciones:

* La primera llamada cantidad_filmaciones_mes, se le entrega un mes en español y devuelve la cantidad de películas hechas en ese mes.

* La segunda llamada cantidad_filmaciones_dia, se le entrega un día en español y devuelve la cantidad de películas hechas en ese día.

* La tercera llamda score_titulo, se ingresa un título de película y devuelve el mismo con su año de estreno y su popularidad.

* La cuarta llamada votos_titulo, se ingresa un título de película y devuelve el mismo con el año de estreno, la cantidad de votos y su promedio. En caso de que los votos sean menores a 2000, devuelve un mensaje que dice que la cantidad de votos es insuficiente.

* La quinta llamada get_actor, se ingresa el nombre del actor y se devuelve su nombre, la cantidad de película en las que participó, su retorno y un promedio del retorno.

* La sexta llamada get_director, se ingresa el nombre del director y se devuelve su nombre, el retorno del mismo y las película en las que participó, a demás de los datos de cada una.

* La séptima llamada recomendacion, se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.

## Analisis Exploratiorio de los Datos (EDA)
  * Para saber un poco más de los datos y su comportamiento se procedió a crear un EDA, en él se analizaron los dos dataset.

## Sistema de recomendacion
  * Entrenamos un modelo de machine learning para crear un sistema de recomendación de películas basado en similitud de contenido. Este sistema recomienda películas similares a una dada. Entrenamos el modelo utilizando TF-IDF y similitud de coseno.
