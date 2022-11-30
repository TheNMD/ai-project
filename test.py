import numpy as np
import tensorflow as tf

data = np.array([5, 6, 7])
data = tf.keras.utils.normalize(data, axis=-1, order=2)

print(data)