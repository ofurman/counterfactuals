from scipy.spatial.distance import cdist

dist = cdist([[]], [[]], metric="jaccard").diagonal()
print(dist)


dist = cdist([[]], [[]], metric="hamming").diagonal()
print(dist)
