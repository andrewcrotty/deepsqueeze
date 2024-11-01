# DeepSqueeze

This repository contains a basic implementation of DeepSqueeze from our [SIGMOD 2020 paper](https://dl.acm.org/doi/10.1145/3318464.3389734). DeepSqueeze is a semantic compression framework that uses autoencoders to capture complex relationships in real-world tabular datasets, in particular focusing on error-bounded lossy compression of numerical data.

This implementation does not include some advanced techniques described in the paper, such as parameter sharing for categorical attributes, gated mixture of experts, and Bayesian optimization for hyperparameter tuning. If you are interested in these aspects of the work, please see this (unaffiliated) implementation: https://github.com/MikeXydas/DeepSqueeze

You can use the following commands to reproduce the results from the paper:
```
# Corel
python3 deepsqueeze.py data/corel.csv -c brotli -l 11 -e 0.005 -E 1500
python3 deepsqueeze.py corel.csv.tar.gz -d -C data/corel.csv

# Forest
python3 deepsqueeze.py data/forest.csv -c brotli -l 11 -e 0.005 -E 300
python3 deepsqueeze.py forest.csv.tar.gz -d -C data/forest.csv
```

If you use this code, please cite:
```
@inproceedings{DBLP:conf/sigmod/IlkhechiCGMFSC20,
  author       = {Amir Ilkhechi and
                  Andrew Crotty and
                  Alex Galakatos and
                  Yicong Mao and
                  Grace Fan and
                  Xiran Shi and
                  Ugur {\c{C}}etintemel},
  editor       = {David Maier and
                  Rachel Pottinger and
                  AnHai Doan and
                  Wang{-}Chiew Tan and
                  Abdussalam Alawini and
                  Hung Q. Ngo},
  title        = {DeepSqueeze: Deep Semantic Compression for Tabular Data},
  booktitle    = {Proceedings of the 2020 International Conference on Management of
                  Data, {SIGMOD} Conference 2020, online conference [Portland, OR, USA],
                  June 14-19, 2020},
  pages        = {1733--1746},
  publisher    = {{ACM}},
  year         = {2020},
  url          = {https://doi.org/10.1145/3318464.3389734},
  doi          = {10.1145/3318464.3389734},
  timestamp    = {Wed, 04 May 2022 13:02:28 +0200},
  biburl       = {https://dblp.org/rec/conf/sigmod/IlkhechiCGMFSC20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
