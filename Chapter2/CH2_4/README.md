

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-30 20:41:58
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-01 14:57:22
 * @FilePath     : /Deep_Learning/Chapter2/CH2_4/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
<!-- TOC -->

- [Chapter 2 Introduction to Neural Networks](#chapter-2-introduction-to-neural-networks)
    - [2.4 Neural Networks](#24-neural-networks)
        - [2.4.1 multi layers and introduction](#241-multi-layers-and-introduction)
        - [2.4.2 Activation Function](#242-activation-function)
        - [2.4.3 Step function in .ipython](#243-step-function-in-ipython)
        - [2.4.4 Sigmoid function in .ipython](#244-sigmoid-function-in-ipython)
        - [2.4.5 ReLU function in .ipython](#245-relu-function-in-ipython)

<!-- /TOC -->
# Chapter 2 Introduction to Neural Networks

## 2.4 Neural Networks

### 2.4.1 multi layers and introduction
* Layers  
    * input layer

    * hidden layer

    * output layer

* Perceptron  
    * y = 0, if b+w1x1+w2x2 <= 0
    * y = 1, if b+w1x1+w2x2 >  0

    * We can input a **new function** to rewrite this upper function. 
        *  y = h(b+w1x1+w2x2)
            * h(x) = 0, if x<=0
            * h(x) = 1, if x>0
        
    * We call this new function h(x) **activation function**

### 2.4.2 Activation Function

* Step function -> for perceptron  
    In perceptron, the signal is always 0 or 1.

* Sigmoid function -> for neural networks
    In neural networks the signal is **smooth and continuous**.

* The activation function must be **nonlinear functions**. 

* ReLU(Rectified Linear Unit) function:
    * h(x) = x, if x > 0
    * h(x) = 0, if x > 0


### 2.4.3 Step function in .ipython

### 2.4.4 Sigmoid function in .ipython

### 2.4.5 ReLU function in .ipython


