# GDCMAD: Graph-based Dual-Contrastive Representation Learning for Multivariate Time Series Anomaly Detection
## Introduction

A Python implementation of the clustering algorithm presented in:

   <b><i>Sheng He#, Wenxuan He#, Mingjing Du*, Xiang Jiang, Yongquan Dong*. GDCMAD: Graph-based dual-contrastive representation learning for multivariate time series anomaly detection. <i> Information Sciences</i>, 2026, 728: 122790.</i></b>

The paper is available online at: <a href="https://www.sciencedirect.com/science/article/abs/pii/S1566253524005906" target="_blank">pdf</a>. 

If you use this implementation in your work, please add a reference/citation to the paper. You can use the following bibtex entry:

```
@article{HeHDJD26,
  author       = {Sheng He and
                  Wenxuan He and
                  Mingjing Du and
                  Xiang Jiang and
                  Yongquan Dong},
  title        = {GDCMAD:Graph-based dual-contrastive representation learning for
                  multivariate time series anomaly detection},
  journal      = {Information Sciences},
  volume       = {728},
  pages        = {122790},
  year         = {2026},
  doi          = {10.1016/J.INS.2025.122790},
}
```

## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)

# Dataset

* SMAP and MSL:

```
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

* SMD:

```
https://github.com/NetManAIOps/OmniAnomaly
```

* SWaT:

```
http://itrust.sutd.edu.sg/research/dataset
```


* Run the code

```
python main.py <dataset>
```

where `<dataset>` is one of `SMAP`, `MSL`, `SMD`, `SWAT`, `PSM`, `ASD`

For more related researches, please visit my homepage: https://dumingjing.github.io/. For data and discussion, please message Mingjing Du (杜明晶@江苏师范大学): dumj@jsnu.edu.cn.

