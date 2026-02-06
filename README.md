# YOLOv11 para Monitoreo de Insectos Sociales

En este repositorio se encuentra el código desarrollado para la detección y el seguimiento de insectos sociales utilizando el modelo **YOLOv11**.

El objetivo principal del proyecto es **inferir el tipo de avispa presente en cada frame de video y realizar el tracking de cada insecto**, con el fin de analizar y estudiar el comportamiento de la especie a lo largo del tiempo.

Todas las actividades, dudas y experimentos realizados durante el desarrollo del proyecto fueron documentados en la siguiente bitácora:
https://docs.google.com/document/d/1_7aP9gireAEJLyceqpsj3g371sJiPpdeGVt_UWBfflw/edit?usp=sharing 

## Integrantes: 
* [Nataly Hofkamp](https://github.com/NatalyHofkamp)
* [Candela Castillo](https://github.com/castillocande)


## Dataset

El dataset fue anotado y generado utilizando la plataforma **Roboflow**, mediante la anotación manual de **1.740 imágenes**, que posteriormente fueron aumentadas hasta un total de **4.176 imágenes** aplicando distintas técnicas de *data augmentation*.

Para la discriminación entre castas se utilizaron las siguientes etiquetas:
- **avispa** = 0  
- **reina** = 1  
- **zángano** = 2  

Se mantuvo la misma división del dataset utilizada en modelos anteriores, con la siguiente proporción:
- **88%** entrenamiento  
- **8%** validación  
- **4%** test  

---

### Consideraciones

El modelo **YOLOv8** fue entrenado utilizando imágenes aumentadas. Se intentó implementar un método que permitiera distinguir entre imágenes originales y aumentadas con el objetivo de entrenar el nuevo modelo únicamente con imágenes originales; sin embargo, no fue posible encontrar un criterio exacto y confiable para realizar esta separación.

Dado que algunas técnicas de *augmentation* generan imágenes tipo mosaico compuestas por múltiples subimágenes, surge la hipótesis de que este tipo de transformaciones podría estar introduciendo ruido en el entrenamiento. En particular, en varias imágenes aparecen tornillos u otros elementos en las esquinas que, tras la transformación, podrían ser interpretados erróneamente por el modelo como avispas.

---

### Entrenamiento

Para el entrenamiento del modelo se utilizaron los siguientes hiperparámetros:
- *Learning rate*: **0.015**
- *Batch size*: **4**
- *Epochs*: **150**

 

## Métricas 
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/d225e70d-0ee5-45d1-8f87-e4190e7a5fec" />

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/50f7be40-c38d-487d-9659-9bfbd99753d6" />

Es posible notar una mejora respecto a los resultados de accuracy del modelo YOLOv8, logrando predecir de manera correcta el dataset [avispa] un 99%, [reina] un 97% y [zangano] un 99%.

## Distribución de los archivos

### `src/`
- **models/**  
  Carpeta que contiene las distintas versiones entrenadas del modelo base (*yolo11n.pt*), incluyendo los pesos generados durante el entrenamiento (*best.pt*, *last.pt*).

- **training_v11.py**  
  Script principal para el entrenamiento del modelo YOLOv11.  
  Permite especificar por argumentos los hiperparámetros utilizados, tales como:
  - modelo base
  - ruta del dataset
  - *learning rate*
  - número de *epochs*
  - tamaño de imagen (*image size*)

---

### `scripts/`
- **detect_mosaic.py**  
  Crea las carpetas `augmented_images` y `not_augmented_images`.  
  Este script corresponde a un intento de detección automática de imágenes aumentadas dentro del dataset.  
  En la práctica no se obtuvieron resultados satisfactorios, incluso incorporando técnicas como *Otsu* y *CLAHE* para mejorar la detección, por lo que este enfoque fue finalmente descartado.

- **visualize_predictions.py**  
  Genera las carpetas `correct_images` e `incorrect_images`, separando las imágenes con predicciones correctas y erróneas.  
  Resalta las avispas detectadas mediante *bounding boxes* de distinto color según el origen de la predicción:
  - **Predicciones del modelo**  
    - rojo: avispa  
    - azul: reina  
    - amarillo: zángano  
  - **Etiquetas provistas por los investigadores**  
    - rosa: avispa  
    - celeste: reina  
    - coral: zángano  

  Además, muestra en pantalla la cantidad de falsos positivos y falsos negativos detectados por imagen.

- **track_wasps.py**  
  Realiza el seguimiento (*tracking*) del recorrido de las avispas en un video utilizando el modelo YOLOv11 para obtener las predicciones.  
  Dado que inicialmente se contaba con un único video de avispas en movimiento, este se utiliza como caso de prueba.

  **Observaciones:**
  - El modelo logra realizar un tracking consistente de los insectos, aunque en algunos frames puede asignar etiquetas diferentes al mismo individuo.
  - Al ejecutar el script, se muestra el video con el tracking en tiempo real.
  - Para detener la ejecución, presionar la tecla **`Q`** desde la consola.

---

### `results/`
- **prediction_visualization/**  
  Contiene las imágenes clasificadas según las predicciones generadas por `visualize_predictions.py`.

- **v11_150ep_0015lr/**  
  Incluye los archivos generados automáticamente por Ultralytics al finalizar el entrenamiento del modelo.

---

### `data/`
Conjunto de datos provisto para el análisis y entrenamiento del modelo.

- **images/**
  - `train/`
  - `valid/`
  - `test/`

- **videos/**


## Observaciones y mejoras

- Es posible realizar nuevos entrenamientos variando los hiperparámetros con el objetivo de analizar su impacto en el rendimiento del modelo y detectar posibles mejoras.
- Se encuentran disponibles cientos de videos adicionales para el análisis del comportamiento de avispas, que pueden utilizarse como conjunto de test y para validar el correcto funcionamiento del algoritmo de seguimiento ByteTrack.
- Implementar un método que permita diferenciar las imágenes aumentadas de las originales, con el fin de poder eliminarlas selectivamente y evaluar otras técnicas de data augmentation que contribuyan a mejorar los resultados del entrenamiento.
- Evaluar si el entrenamiento del modelo incorporando las dimensiones de cada *bounding box* como características adicionales mejora la performance de la clasificación.


