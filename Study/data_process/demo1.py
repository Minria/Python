import gzip
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
path = 'D:/wangfuming/Desktop/mnist.pkl.gz'
with gzip.open(path, 'rb') as f:
    a = pickle.load(f, encoding='latin1')
    (x_train, y_train), (x_test, y_test), (_, _) = pickle.load(f, encoding='latin1')
    demo = x_train.reshape(-1,28,28)
    img = demo[0]*255
    im = Image.fromarray(img)
    im.show()
    # plt.imshow(img)
    # plt.show()
    print(1)