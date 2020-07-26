

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-29 16:55:08
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-26 20:59:56
 * @FilePath     : /undefined/home/jingsheng/Deep_Learning/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

# Deep_Learning
A repository about Deep_Learning

## Content

- [Chapter 1 Math and NumPy](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1)
    - [1.1 Data Dimension](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2)
    - [1.2 Data in Numpy](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
        - [1.2.1 Scalars](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
        - [1.2.2 Vectors](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
        - [1.2.3 Matrices](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
        - [1.2.4 Tensors](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
        - [1.2.5 Reshape](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
        - [1.2.6 Broadcast](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_1And2#12-data-in-numpy)
    - [1.3 Matrix Operation](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_3)
    - [1.4 Multiplying the matrices](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_4)
        - [1.4.1 Multiplying in NumPy:](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_4)
        - [1.4.2 Dot Product](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_4)
    - [1.5 Matrix transpose](https://github.com/jingshenglyu/Deep_Learning/tree/master/Chapter1/CH1_5)
    - [1.6 NumPy Test](https://github.com/jingshenglyu/Deep_Learning/blob/master/Chapter1/CH1_6/NumPy_Test.ipynb)

- [Chapter 2 Introduction to Neural Networks](#chapter-2-introduction-to-neural-networks)
    - [2.1 Perceptron](#21-perceptron)
    - [2.2 Logical Connective](#22-logical-connective)
    - [2.3 Linear vs Nonlinear](#23-linear-vs-nonlinear)
        - [2.3.1 Multi-perceptron](#231-multi-perceptron)
        - [2.3.2 Linear vs Nonlinear](#232-linear-vs-nonlinear)
    - [2.4 Neural Networks](#24-neural-networks)
        - [2.4.1 multi layers and introduction](#241-multi-layers-and-introduction)
        - [2.4.2 Activation Function](#242-activation-function)
    - [2.5 Neural Networks](#25-neural-networks)
        - [2.5.1 Dot Product](#251-dot-product)
        - [2.5.2 Three Layers Neural Networks](#252-three-layers-neural-networks)
        - [2.5.3 Activation Function for Output-Layer](#253-activation-function-for-output-layer)
    - [2.6 Practice](#26-practice)
        - [2.6.1 Handwritten Numeral Recognition](#261-handwritten-numeral-recognition)
        - [2.6.2 Handwritten Numeral Recognition by Batch Data](#262-handwritten-numeral-recognition-by-batch-data)
            - [load_mnist() function:](#load_mnist-function)
            - [Neural Network for MNIST](#neural-network-for-mnist)

- [Chapter 3 Data Learning in Neural Network](#chapter-3-data-learning-in-neural-network)
    - [3.1 feature](#31-feature)
    - [3.2 Loss Function](#32-loss-function)
        - [3.2.1 **Generalization** ability:](#321-generalization-ability)
        - [3.2.2 Over fitting:](#322-over-fitting)
        - [3.2.3 Loss(Cost) Function](#323-losscost-function)
    - [3.3 Mini-Batch](#33-mini-batch)
    - [3.4 Numerical Differentiation](#34-numerical-differentiation)
        - [3.4.1 Derivative:](#341-derivative)
        - [3.4.2 Partial Derivative](#342-partial-derivative)
        - [3.4.3 Gradient](#343-gradient)
    - [3.5 Gradient Method](#35-gradient-method)
        - [3.5.1 gradient **descent** method](#351-gradient-descent-method)
        - [3.5.2 gradient **ascent** method](#352-gradient-ascent-method)
        - [3.5.3 gradient for neural network](#353-gradient-for-neural-network)
    - [3.6 Gradient of neural network](#36-gradient-of-neural-network)

- [Chapter 4 Error Back Propagation](#chapter-4-error-back-propagation)
    - [4.1 Basic Content](#41-basic-content)
    - [4.1.1 Numerical Differentiation vs Error Back Propagation](#411-numerical-differentiation-vs-error-back-propagation)
    - [4.2 Forward Propagation and Backward Propagation](#42-forward-propagation-and-backward-propagation)
        - [4.2.1 Forward Propagation(FP)](#421-forward-propagationfp)
        - [4.2.2 Backward Propagation(BP)](#422-backward-propagationbp)
    - [4.3 Derivative](#43-derivative)
        - [4.3.1 Chain Rule](#431-chain-rule)
    - [4.4 Activation Function for Computational Graph of Neural Network](#44-activation-function-for-computational-graph-of-neural-network)
        - [4.4.1 Activation Function](#441-activation-function)
        - [4.4.2 Affine Layer](#442-affine-layer)
    - [4.5 Gradient Check](#45-gradient-check)
- [Chapter 5](#chapter-5)
    - [5.1 Methode of the update of the parameters(weights and bias)](#51-methode-of-the-update-of-the-parametersweights-and-bias)
        - [5.1.1 SGD](#511-sgd)
        - [5.1.2 Momentum](#512-momentum)
        - [5.1.3 AdaGrad](#513-adagrad)
        - [5.1.4 Adam](#514-adam)
    - [5.2 Methode of the assignment for the initial values of the weights](#52-methode-of-the-assignment-for-the-initial-values-of-the-weights)
    - [5.3 Batch Normalization](#53-batch-normalization)
    - [5.4 Droput](#54-droput)
- [Chapter6 Convolution Neural Network](#chapter6-convolution-neural-network)
    - [6.1 Structure](#61-structure)
    - [6.2 CNN Layer](#62-cnn-layer)
        - [6.2.1 Fully Connection vs Local Connection](#621-fully-connection-vs-local-connection)
        - [6.2.2 Feature Map](#622-feature-map)
        - [6.2.3 Convolution/Filter Operation](#623-convolutionfilter-operation)
        - [6.2.4 Weights and Bias for CNN](#624-weights-and-bias-for-cnn)
        - [6.2.5 Padding and Stride](#625-padding-and-stride)
        - [6.2.6 n-dim for CNN](#626-n-dim-for-cnn)
        - [Chapter6 Convolution Neural Network](#chapter6-convolution-neural-network)
    - [6.3 Pooling Layer](#63-pooling-layer)
        - [6.3.1 Purpose:](#631-purpose)
        - [6.3.2 Characters of Pooling Layer](#632-characters-of-pooling-layer)
        - [6.3.3 Base on im2col](#633-base-on-im2col)
    - [6.4 Code for CNN and Pooling](#64-code-for-cnn-and-pooling)
        - [6.4.1 CNN](#641-cnn)
        - [6.4.2 Pooling](#642-pooling)


# Reference
1. Ian Goodfellow and Yoshua Bengio and Aaron Courville, An MIT Press book， Deep Learning. [in English](http://www.deeplearningbook.org/), [PDF](https://github.com/janishar/mit-deep-learning-book-pdf/blob/master/complete-book-pdf/Ian%20Goodfellow%2C%20Yoshua%20Bengio%2C%20Aaron%20Courville%20-%20Deep%20Learning%20(2017%2C%20MIT).pdf) [in Chinese](https://github.com/exacity/deeplearningbook-chinese)

2. Andrew W. Trask, Grokking Deep Learning, published in 2019 [PDF](http://www.hdip-data-analytics.com/_media/resources/pdf/s4/grokking_deep_learning.pdf)

3. Michael Nielsen, Neural Networks and Deep Learning, published in 2019 [PDF](http://static.latexstudio.net/article/2018/0912/neuralnetworksanddeeplearning.pdf)

4. The illustration deep learning [in Chinese](https://github.com/IammyselfYBX/The_illustration_deep_learning/blob/master/BOOKS/%E5%9B%BE%E8%A7%A3%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0.pdf)

5.  Jon Krohn, Grant Beyleveld and Aglaé Bassens, Deep Learning Illustrated, published in 2020 [Repository](https://github.com/the-deep-learners/deep-learning-illustrated)

6. O'Reilly Japan, Deep Learning from Scratch in 2019. [in Chinese](https://github.com/LeoLiu8023AmyLu/Machine_Learning/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8-%E5%9F%BA%E4%BA%8Epython%E7%9A%84%E7%90%86%E8%AE%BA%E4%B8%8E%E5%AE%9E%E7%8E%B0.pdf)  
[Repository in Japanese](https://github.com/oreilly-japan/deep-learning-from-scratch)

# Tools:
1. [PyTorch](https://github.com/ZhiqiangHo/awesome-machine-learning/blob/master/Pytorch%20%E3%80%8A%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%E4%B9%8BPyTorch.%E5%BB%96%E6%98%9F%E5%AE%87(%E8%AF%A6%E7%BB%86%E4%B9%A6%E7%AD%BE)%E3%80%8B.pdf)



