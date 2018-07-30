
import numpy as np
import pandas as pd
import time
import faiss
#import _swigfaiss_gpu
import sys
k     = 10
ngpu  = 1
def load_data():
  df = pd.read_csv('../df.csv')
  data = np.ascontiguousarray(df.values)
  print(data.shape)
  return data

x = load_data() 
print("reshape")
x = x.reshape(x.shape[0], -1).astype('float32')
def train_kmeans(x, k, ngpu):
    "Runs kmeans on one or several GPUs"
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter   = 20
    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000
    res = [faiss.StandardGpuResources() for i in range(ngpu)]
    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)
    if ngpu == 1:
        index = faiss.GpuIndexFlatIP(res[-1], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatIP(res[i], d, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)
    # perform the training
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    obj = faiss.vector_float_to_array(clus.obj)
    print("final objective: %.4g" % obj[-1])
    return centroids.reshape(k, d)
print("run")
t0 = time.time()
ret = train_kmeans(x, k, ngpu).tolist()
print(ret)
import json
json.dump(ret, fp=open('ip_kmeans.json', 'w'))
t1 = time.time()
print("total runtime: %.3f s" % (t1 - t0))
