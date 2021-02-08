import ssl
# import numpy as np
# import urllib
# import PIL.Image
import tensorflow as tf
# import tensorflow_docs.plots
# import tensorflow_docs as tfdocs
import tensorflow_datasets as tfds
from src.module_aumentation import visualize

AUTOTUNE = tf.data.experimental.AUTOTUNE

# para https urls
ssl._create_default_https_context = ssl._create_unverified_context

# solo sacar 10 ejemplos de entrenamiento
ds, ds_info = tfds.load('colorectal_histology', split='train',
                        shuffle_files=True, with_info=True, download=False)
assert isinstance(ds, tf.data.Dataset)
print(ds_info)

# ver imagenes
fig = tfds.show_examples(ds_info, ds)

# leer todas las imagenes(break cuando las lee todas)
for example in tfds.as_numpy(ds):
    image, label = example['image'], example['label']
    break

# tomar otro ejemplo de la data
one_sample = ds.take(1)
one_sample = list(one_sample.as_numpy_iterator())
image = one_sample[0]['image']
label = one_sample[0]['label']
print(image.shape, label.shape)

# ajustar brillo con tf
bright = tf.image.adjust_brightness(image, 0.2)
visualize(image, bright, 'brightened image')

# Flip de la imagen
flipped = tf.image.flip_left_right(image)
visualize(image, flipped, 'flipped image')
# ajustada
adjusted = tf.image.adjust_jpeg_quality(image, jpeg_quality=20)
visualize(image, adjusted, 'quality adjusted image')

# hacer un random crop de la imagen
crop_to_original_ratio = 0.5
new_size = int(crop_to_original_ratio * image.shape[0])
cropped = tf.image.random_crop(image, size=[new_size, new_size, 3])
visualize(image, cropped, 'randomly cropped image')

# Recorte central de la imagen (el área de recorte está en el centro)
central_fraction = 0.6
center_cropped = tf.image.central_crop(
    image, central_fraction=central_fraction)
visualize(image, center_cropped, 'centrally cropped image')

# agregar ruido gausiano
common_type = tf.float32
gnoise = tf.random.normal(
    shape=tf.shape(image), mean=0.0, stddev=0.1, dtype=common_type)
image_type_converted = tf.image.convert_image_dtype(
    image, dtype=common_type, saturate=False)
noisy_image = tf.add(image_type_converted, gnoise)
visualize(image_type_converted, noisy_image, 'noisyimage')
