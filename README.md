# data-augmentation

Se usa texturas en el conjunto de datos de histología del cáncer colorrectal 
- (ref: Tensorflow Dataset: https://www.tensorflow.org/datasets/catalog/colorectal_histology.
- Cada imagen es de tamaño 150 x 150 x 3 RGB de 8 clases diferentes y hay 5000 imágenes.

Algunas funciones de uso frecuente:
* split =: elija la división predefinida para leer (leanse la [guía de la API de TensorFlow] (https://www.tensorflow.org/datasets/splits)).
* shuffle_files =: Mezcla los archivos en cada época si es "Verdadero". Esto crea diferentes lotes para cada época de entrenamiento.
* data_dir =: Ubicación de los datos guardados (predeterminado: ~ / tensorflow_datasets /)
* with_info =: Devuelve tfds.core.DatasetInfo que contiene metadatos del conjunto de datos
* download =: Si es "True", descarga el conjunto de datos. Una vez que lo descargamos, para futuras llamadas, lo configuramos como "Falso", ya que no es necesario volver a descargarlo.

De donde saqué todo [aquí] (https://www.tensorflow.org/datasets/api_docs/python/tfds/load)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

### SETUP Pasos a seguir para usar la aumentación de imagenes

```sh
$ git clone https://github.com/matheus695p/data-augmentation.git
$ cd data-augmentation
$ echo instalar los requirements
$ pip install -q tensorflow-datasets
$ pip install -q git+https://github.com/tensorflow/docs
$ pip install -r requirements.txt
```
