import numpy as np
import tensorflow as tf

data = np.array([5., 6., 7.], dtype='float32')
min_max =  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

print(min_max(data))