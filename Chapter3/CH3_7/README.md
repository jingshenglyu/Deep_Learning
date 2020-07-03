

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-04 00:11:56
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-04 00:34:16
 * @FilePath     : /Deep_Learning/Chapter3/CH3_7/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

# Chapter 3 Data Learning in Neural Network

## 3.7 How to make a learning argorithm of neural network?

0. What is a learning of neural network?
    * Learning:  
        Adjusting the weights and biases to fit the training data. The process is called "learning."
1. mini-batch:  
    * A randomly selected portion of the training data is called the mini-batch.
Our goal is to **reduce the value of the loss functio**n of mini-batch.

2. Calculate the gradient
    * In order to reduce the value of the loss function of mini-batch, the gradient of each weight parameter is required. The gradient indicates **the direction in which the value of the loss function decreases the most.**

3. Update the weights
    * The weighting parameters are updated slightly along the gradient direction.

4. Repeat