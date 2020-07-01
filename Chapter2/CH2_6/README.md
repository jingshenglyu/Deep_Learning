

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-01 20:41:12
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-01 23:41:35
 * @FilePath     : /Deep_Learning/Chapter2/CH2_6/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
* !!! For this folder, if you want to use these followed code. You must ***download this complett*** folder. Because the **dataset** and **the function that we will use** are included in this folder. !!! 
<!-- TOC -->

- [Chapter 2 Introduction to Neural Networks](#chapter-2-introduction-to-neural-networks)
    - [2.6 Practice](#26-practice)
        - [2.6.1 Handwritten Numeral Recognition](#261-handwritten-numeral-recognition)
        - [2.6.2 Handwritten Numeral Recognition by Batch Data](#262-handwritten-numeral-recognition-by-batch-data)
            - [load_mnist() function:](#load_mnist-function)
            - [Neural Network for MNIST](#neural-network-for-mnist)

<!-- /TOC -->
# Chapter 2 Introduction to Neural Networks

## 2.6 Practice
### 2.6.1 Handwritten Numeral Recognition
[Code is here](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter2/CH2_6/MNIST_dataset.ipynb)

### 2.6.2 Handwritten Numeral Recognition by Batch Data
[Code is here](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter2/CH2_6/Batch.ipynb)

#### load_mnist() function:
* It will return (x_train, t_train) (x_test, t_test). 
    * x: image
    * t: test
* load_mnist(normalize=True, flatten=True, one_hot_label=False)
    * normalize
        * `= True`: 0.0 ~ 1.0
        * `= False`: 0 ~ 255
        
    * flatten
        * `= True`: 1-dim array with 784 pixels
        * `= False`: 3-dim arrays with 1x28x28
        
    * one_hot_label
        * `= True`: The label is one-hot.
        * `= False`: normal

#### Neural Network for MNIST

* Input-Layer:   
    784 Nodes(pixel) = 28*28, because the image from the dataset of MNIST is always 28*28.

* Output-Layer:   
    10, because there are 10 numbers for decimalism.

* Hidden-Layer:  
    2 layers. The number of the nodes from these Hidden Layer is not important. We set here for 1st Hidden-Layer to 50, 2nd Hidden-Layer to 100. We can also set other number.