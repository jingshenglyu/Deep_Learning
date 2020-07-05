

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-02 21:55:41
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-04 00:08:07
 * @FilePath     : /Deep_Learning/Chapter3/CH3_5/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
<!-- TOC -->

- [Chapter 3 Data Learning in Neural Network](#chapter-3-data-learning-in-neural-network)
    - [3.5 Gradient Method](#35-gradient-method)
        - [3.5.1 gradient **descent** method](#351-gradient-descent-method)
        - [3.5.2 gradient **ascent** method](#352-gradient-ascent-method)
        - [3.5.3 gradient for neural network](#353-gradient-for-neural-network)

<!-- /TOC -->
# Chapter 3 Data Learning in Neural Network

## 3.5 Gradient Method
### 3.5.1 gradient **descent** method
* formula:

    * x0 = x0 - $\eta$ * (df/dx0)
    * x1 = x1 - $\eta$ * (df/dx1)

* learning rate
    * $\eta$, it decide to do how much does the neural network learn. It is also called **hyperparamer**. *Weights* are from training data and got. **Learning rate is designed by designer.**


### 3.5.2 gradient **ascent** method
* Find the best parameters(weights and bias), that means the loss function has the minimal values.

### 3.5.3 gradient for neural network
* find the maximal values