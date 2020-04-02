import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# generate 2d classification dataset
X, Y = make_blobs(n_samples=1000, centers=2, n_features=2)
x, x_val, y, y_val = train_test_split(X, Y, test_size=0.2)

# init parameters
x_shape = x.shape
w = np.random.normal(0, 1, x_shape[1])
b = np.random.normal(0, 1, 1)
lr = 0.01
epoch = 1000

# sigmoid function
def sig(x):
    return (1.0 / (1.0 + np.exp(-x)))

def get_accuracy(w, b, x_val, y_val, th=0.5):
    z = (x_val * w).sum(axis=1) + b
    a = sig(z)
    y_pred = np.where(a >= 0.5, 1, 0)
    total_true = (y_pred == y_val).astype(int).sum()
    return total_true / x_val.shape[0]

for e in range(epoch):
    z = (x * w).sum(axis=1) + b
    a = sig(z)
    loss = y * -(np.log(a)) - (1.0 - y) * np.log(1.0 - a)

    # update weights
    dl_da = (-y / a) + (1.0 - y) / (1.0 - a)
    da_dz = a * (1 - a)
    dz_dw = x
    dl_dz = np.tile(dl_da * da_dz, (x_shape[1], 1)).transpose()
    dl_dw = dl_dz * dz_dw
    w = w - (lr * dl_dw.mean(axis=0))

    # update bias
    dl_de = dl_da * da_dz
    b = b - (lr * dl_de.mean(axis=0))
    
    print(loss.mean(), get_accuracy(w, b, x_val, y_val))
