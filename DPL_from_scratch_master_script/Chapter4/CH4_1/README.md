

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-04 08:41:54
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-04 09:34:23
 * @FilePath     : /Deep_Learning/Chapter4/CH4_1/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter 4 Error Back Propagation](#chapter-4-error-back-propagation)
    - [4.1 Basic Content](#41-basic-content)
    - [4.1.1 Numerical Differentiation vs Error Back Propagation](#411-numerical-differentiation-vs-error-back-propagation)

<!-- /TOC -->
# Chapter 4 Error Back Propagation

## 4.1 Basic Content

## 4.1.1 Numerical Differentiation vs Error Back Propagation
* Numerical Differentiation:    
    Easily to realize, but it **costs too much time** for calculation.
* Error Back Propagation:   
    Efficiently to **calculate the gradient of weights**

    * Base on mathematical expression
    * Base on **computational graph**

 ### 4.1.2 computational graph
* Advantage:  
    * partial calculation
    * save the intermediate result
    * **Computational grapf can be used by inverse and efficiently computation of derivatives for directional propagation.**

