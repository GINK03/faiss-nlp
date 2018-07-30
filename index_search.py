import faiss
import numpy as np
import pandas as pd
import time
import sys
import pickle
n_candidates = 10

arr, voc, urls = pickle.load(open('../arr_voc_url.pkl', 'rb'))

arr = arr.todense()
height, width = arr.shape

if '--faiss' in sys.argv:
  index = faiss.IndexFlatIP(width)
  index.add( np.ascontiguousarray(arr).astype(np.float32) )
  xq = np.ascontiguousarray(arr).astype(np.float32)[10:11]
  r = index.search(xq, n_candidates)
  print(xq)
  print(r)

if '--quantize' in sys.argv:
  nlist = 1000 # 分割粒度
  print('start to quantize')
  quantizer = faiss.IndexFlatL2(width)
  index = faiss.IndexIVFFlat(quantizer, width, nlist, faiss.METRIC_L2)
  index.train( np.ascontiguousarray(arr).astype(np.float32)  )
  print('finished to quantize')

  xq = np.ascontiguousarray(arr).astype(np.float32)[10:11]

  index.nprobe = 10
  r = index.search(xq, n_candidates)
  print(xq)
  print(r)
