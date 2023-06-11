# Importamos librerias
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Ahora leeremos nuestros datos
peliculas = pd.read_csv('peliculas.csv')  # Peliculas


# Los imprimimos en pantalla un resumen
print('Peliculas:\n', peliculas.head())

# Dividir valores de genero en columnas
peliculas['listed_in'] = peliculas.listed_in.str.split(',')

peliculas_co = peliculas.copy()  # Copia
for index, row in peliculas.iterrows():  # Pasamos la matriz original
    for genero in row['listed_in']:  # Se asigna 1 al genere correspondiente
        peliculas_co.at[index, genero] = 1  # Se guarda en la copia

peliculas_co = peliculas_co.fillna(0)  # Asignamos 0 en el resto de generos
print('\nPeliculas codificadas:\n', peliculas_co)

# Crear perfil  del usuario
usuario_en = [
    {'title': 'The Witcher: Nightmare of the Wolf', 'ratingU': 5},
    {'title': 'Deadly Switch', 'ratingU': 3.5},
    {'title': 'Way Back into Love', 'ratingU': 2},
    {'title': 'Accidentally in Love', 'ratingU': 3},
    {'title': 'Dinosaur King', 'ratingU': 4.5}
]

entrada_peli = pd.DataFrame(usuario_en)
print('\nPeliculas Usuario:\n', entrada_peli)

# Agregando ID reales a las peliculas del usuario
Id = peliculas[peliculas['title'].isin(entrada_peli['title'].tolist())]  # Filtrado
entrada_peli = pd.merge(Id, entrada_peli)  # Fusion de matrices

# Eliminar info innecesaria
entrada_peli = entrada_peli.drop('type', axis='columns').drop('director', axis='columns').drop('cast', axis='columns').drop('country', axis='columns').drop('date_added', axis='columns').drop('release_year', axis='columns').drop('rating', axis='columns').drop('duration', axis='columns').drop('description', axis='columns')

# Codificar peliculas de usuario - One Hot Encoder
peli_usuario = peliculas_co[peliculas_co['show_id'].isin(entrada_peli['show_id'].tolist())]
print('\nPeliculas Usuario Codificadas:\n', peli_usuario)

# Eliminar info innecesaria
peli_usuario = peli_usuario.reset_index(drop=True)
tabla_generos = peli_usuario.drop('show_id', axis='columns').drop('title', axis='columns').drop('type', axis='columns').drop('director', axis='columns').drop('cast', axis='columns').drop('country', axis='columns').drop('date_added', axis='columns').drop('release_year', axis='columns').drop('rating', axis='columns').drop('duration', axis='columns').drop('listed_in', axis='columns').drop('description', axis='columns')
print('\nTabla de géneros:\n', tabla_generos)

# Modelar algoritmo.
# Crear la matriz de peso
perfi_usu = tabla_generos.transpose().dot(entrada_peli['ratingU'])
print('\nCategoria que usuario prefiere:\n',perfi_usu)

#Extraer generos de la tabla original
generos = peliculas_co.set_index(peliculas_co['show_id'])

# Mostramos la info necesaria (Generos) Ponderacion de usuario
generos = generos.drop('show_id', axis='columns').drop('title', axis='columns').drop('type', axis='columns').drop('director', axis='columns').drop('cast', axis='columns').drop('country', axis='columns').drop('date_added', axis='columns').drop('release_year', axis='columns').drop('rating', axis='columns').drop('duration', axis='columns').drop('listed_in', axis='columns').drop('description', axis='columns')
print('\nGeneros:\n',generos.head())

#Promedio ponderado para recomendar peliculas
recom = ((generos*perfi_usu).sum(axis=1))/(perfi_usu.sum())
print('\nRecomendaciones:\n',recom.head())

#Orden descendente de las recomandaciones
recom = recom.sort_values(ascending=False)
print('\nRecomendaciones Organizadas\n', recom.head())

#Tabla final
final = peliculas.loc[peliculas['show_id'].isin(recom.head(20).keys())]
nfinal = final[['title']]
print('\nNombres de películas Recomendadas:\n', nfinal)