import numpy as np
from PIL import Image

img_array=np.load('x_test.npy')

im = Image.fromarray(img_array.astype(np.uint8))
#im = Image.fromarray(img_array)
im.show()
