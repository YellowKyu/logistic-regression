import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# generate 2d classification dataset
X, Y = make_blobs(n_samples=1000, n_features=10, centers=5)

# visualization
#plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
#plt.show()

# splitting data
x, x_val, y, y_val = train_test_split(X, Y, test_size=0.2)

# init params
k = 5
rand_index = np.random.choice(x.shape[0], k)
centroids = x[rand_index]

iteration = 0
while True:
    # compute dist between centroids and sample data
    diff = (x[:, np.newaxis] - centroids)
    dist = np.linalg.norm(diff, ord=2, axis=(2))
    dist_min = np.argmin(dist, axis=1)

    # update centroids
    updated = False
    for c in range(k):
        # gather all sample assigned to a category c
        index_all_c = np.where(dist_min == c)
        all_c = x[index_all_c]

        # computer mean of gathered samples and update centroids
        mean_all_c = all_c.mean(axis=0)
        if np.array_equal(mean_all_c, centroids[c]) is False:
            updated = True
        centroids[c] = mean_all_c

    iteration += 1
    print(iteration)
    if updated == False:
        break

diff = (x[:, np.newaxis] - centroids)
dist = np.linalg.norm(diff, ord=2, axis=(2))
y_pred = np.argmin(dist, axis=1)

# visualize real cagetory vs clusters found by k-means over 2 features
fig, axs = plt.subplots(2)
axs[0].scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolor='k')
axs[1].scatter(x[:, 0], x[:, 1], marker='o', c=y_pred, s=25, edgecolor='k')
plt.show()
