

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-04 08:41:46
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-05 23:09:03
 * @FilePath     : /Deep_Learning/Chapter4/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

# Chapter 4 Error Back Propagation

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