

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-29 16:55:08
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-01 23:50:30
 * @FilePath     : /Deep_Learning/README.md
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


# Reference
1. Ian Goodfellow and Yoshua Bengio and Aaron Courville, An MIT Press book， Deep Learning. [in English](http://www.deeplearningbook.org/), [PDF](https://github.com/janishar/mit-deep-learning-book-pdf/blob/master/complete-book-pdf/Ian%20Goodfellow%2C%20Yoshua%20Bengio%2C%20Aaron%20Courville%20-%20Deep%20Learning%20(2017%2C%20MIT).pdf) [in Chinese](https://github.com/exacity/deeplearningbook-chinese)

2. Andrew W. Trask, Grokking Deep Learning, published in 2019 [PDF](http://www.hdip-data-analytics.com/_media/resources/pdf/s4/grokking_deep_learning.pdf)

3. Michael Nielsen, Neural Networks and Deep Learning, published in 2019 [PDF](http://static.latexstudio.net/article/2018/0912/neuralnetworksanddeeplearning.pdf)

4. The illustration deep learning [in Chinese](https://github.com/IammyselfYBX/The_illustration_deep_learning/blob/master/BOOKS/%E5%9B%BE%E8%A7%A3%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0.pdf)

5.  Jon Krohn, Grant Beyleveld and Aglaé Bassens, Deep Learning Illustrated, published in 2020 [Repository](https://github.com/the-deep-learners/deep-learning-illustrated)

6. O'Reilly Japan, Deep Learning from Scratch in 2019. [in Chinese](https://github.com/LeoLiu8023AmyLu/Machine_Learning/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8-%E5%9F%BA%E4%BA%8Epython%E7%9A%84%E7%90%86%E8%AE%BA%E4%B8%8E%E5%AE%9E%E7%8E%B0.pdf)  
[Repository in Japanese](https://github.com/oreilly-japan/deep-learning-from-scratch)



