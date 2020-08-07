

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-08 17:52:15
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-11 10:05:08
 * @FilePath     : /Deep_Learning/Chapter0/PyTorch/Chapter0/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
<!-- TOC -->

- [Chapter 0 Base for PyTorch](#chapter-0-base-for-pytorch)
    - [0.1 Introduction to PyTorch and basic information](#01-introduction-to-pytorch-and-basic-information)
    - [0.2 Storage for Tensor and Stride](#02-storage-for-tensor-and-stride)
        - [0.2.1 Storage for Tensor](#021-storage-for-tensor)
        - [0.2.2 Size, storage_offset and stride](#022-size-storage_offset-and-stride)
    - [0.3 Data Type and Index for PyTorch](#03-data-type-and-index-for-pytorch)
    - [0.4 Numpy <--> PyTorch](#04-numpy----pytorch)
    - [0.5 Serializing for Tensor](#05-serializing-for-tensor)
    - [0.6 Deep Learning on GPU](#06-deep-learning-on-gpu)
    - [0.7 API for PyTorch](#07-api-for-pytorch)

<!-- /TOC -->

# Chapter 0 Base for PyTorch

## 0.1 Introduction to PyTorch and basic information 
* [Code for Base](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_1_base.py)
## 0.2 Storage for Tensor and Stride
### 0.2.1 Storage for Tensor
* How to storage a tensor to a 1-dim array and store it.   
[Code for Storage](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_2_Storage.py)
### 0.2.2 Size, storage_offset and stride
* How to change the size, storage_offset and stride.   
[Code for Size and ...](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_2_Storage.py)
## 0.3 Data Type and Index for PyTorch 
* The basic data type for PyTorch and also the index for Tensor. How to make a slice and other index for the data of a tensor.  
[Code for Data Type and Index](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_3_DataType.py)

## 0.4 Numpy <--> PyTorch
* Interoperability between NumPy and PyTorch. NumPy2PyTorch and PyTorch2NumPy  
[Code for Interoperability](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_4_PyTorch2NumPy.py)

## 0.5 Serializing for Tensor
* How to save and load a tensor's data. We can save them by `.t` or `.pt` file. 
[Code for Serializing](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_5_Serializing.py)

## 0.6 Deep Learning on GPU
* I don't have a PC with the advanced GPU(specially Nvida GPU 6G+). So I will use Google Colab to run the Deep Learning Code on GPU through Google Colab.
* Why to use GPU?
    * Because some training model are very huge. Only CPU is too slow. Using GPU can increase the speed of the training data.

1. You must have a [Google Account](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp) (For example **Gmail**). 
2. You can go into [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) or go there through Google Drive.  

3. With the google notebook(like Jupyter Notebook) you can firstly change the CPU to GPU. 
[Top lists to find **Edit** -> **Notebook settings**]
    * Then here you can change the **runtime type**(Python 3) and **hardware accelerator**(GPU).

4. Now you can run your code on GPU. It will be faster as before. 
    * You can try to write the followed code. If you get an ouput without error, you have already changed it.
        ```
        import torch
        x = torch.empty(5, 3)
        print(x)
        ```
* [Code for GPU](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_6_DPLonGPU.ipynb)

## 0.7 API for PyTorch
* [Short Summary for PyTorch's API](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter0/PyTorch/Chapter0/CH0_7_API_Summary.md)