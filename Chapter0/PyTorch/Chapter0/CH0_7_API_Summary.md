

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-11 08:23:55
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-11 08:34:56
 * @FilePath     : /Deep_Learning/Chapter0/PyTorch/Chapter0/CH0_7_API_Summary.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
<!-- TOC -->

- [0.7 Summary for PyTorch API](#07-summary-for-pytorch-api)
    - [0.7.1 Document from PyTorch](#071-document-from-pytorch)
    - [0.7.2 Summary for API](#072-summary-for-api)

<!-- /TOC -->
# 0.7 Summary for PyTorch API

## 0.7.1 Document from PyTorch
* [API for PyTorch](https://pytorch.org/docs/stable/index.html)


## 0.7.2 Summary for API
1. Create a new tensor:  
    `ones`, `from_numpy`

2. Index, slice and change the shape, stride:  
    `transpose`, `[:, :, ...]`

3. Math:  
    3.1 pointwise:  `abs`, `cos`, ...  
    3.2 reduction: `mean`, `std`, `norm`  
    3.3 compare: `equal`, `max`  
    3.4 frequence: `stft`, `hamming_windows`  
    3.5 others: `cross`, `trace`

4. random sampling:  
    `randn`, `normal`

5. serializing:  
    `save`, `load`
