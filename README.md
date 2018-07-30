# faiss-nlp
faissでNLPで使いそうなやつ

## 用語
 - consine similarity = inner product
 - euclid distance = L2
 
## インストール
コンパイルしていれるのはやめるべき  

anacondaのcondaからインストールする

```console
$ conda install -c pytorch faiss-gpu
```

## innter productでk-meansする

[https://github.com/GINK03/faiss-nlp/blob/master/kmeans-innerproduct.py]

## L2でk-meansする

[https://github.com/GINK03/faiss-nlp/blob/master/kmeans-l2.py]

## faissでgreedyにinnter product距離で計算する

[https://github.com/GINK03/faiss-nlp/blob/master/index_search.py]

`python3 index_search.py --faiss`

## faissでquantizeして距離計算する
（不調、量子化がうまく行っていない模様）

[https://github.com/GINK03/faiss-nlp/blob/master/index_search.py]

`python3 index_search.py --quantize`
