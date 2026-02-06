# YOLOv11 para Monitoreo de Insectos Sociales
En el siguiente repositorio se encuentra el código para la detección de insectos sociales utilizando el modelo YOLOv11. EL propósito de este proyecto es poder inferir el tipo de avispas que se encuentra en cada frame y poder formar un trackeo de cada individuo, con el fin de tener un estudio sobre el comportamiento de la especie. 

* [Nataly Hofkamp](https://github.com/NatalyHofkamp)
* [Candela Castillo](https://github.com/castillocande)


# Dataset 
El dataset fue anotado y generado en la plataforma Roboflow anotando manualmente 1740 imágenes, aumentada a 4176 imágenes con distintas transformaciones. Se utilizaron las etiquetas de [avispa = 0], [zangano = 2] y [reina = 1] para discriminación entre castas. Se mantuvo la división del dataset para el nuevo modelo al igual que con el anterior,88%-8%-4% para sets de entrenamiento-validación-test.
 - consideración: el modelo YOLOv8 fue entrenado con imagenes aumentadas. Se buscó la manera de poder distinguir entre imagenes originales y aumentadas para poder entrenar al nuevo modelo solo con imagenes originales, pero no fue posible encontrar una manera exacta de hacerlo. Al ser un mosaico compuesto por varias imagenes, nos queda la duda sobre si puede estar confundiendo más al modelo, ya que en muchas imagenes aparen tornillos en esquinas que con la versión transformada, puede ser tomado como una avispa.
 - Se  encuentran disponibles cientos de videos para análisis del comportamiento de avispas, que funcionarán como parte del set de Test.

# Métricas 
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/d225e70d-0ee5-45d1-8f87-e4190e7a5fec" />

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/6a8f171c-8cd4-4d3d-92a3-1cde262cba4f" />


