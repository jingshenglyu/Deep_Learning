

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-04 08:41:57
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-05 19:38:59
 * @FilePath     : /Deep_Learning/Chapter4/CH4_2/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter 4 Error Back Propagation](#chapter-4-error-back-propagation)
    - [4.2 Forward Propagation and Backward Propagation](#42-forward-propagation-and-backward-propagation)
        - [4.2.1 Forward Propagation(FP)](#421-forward-propagationfp)
        - [4.2.2 Backward Propagation(BP)](#422-backward-propagationbp)

<!-- /TOC -->

# Chapter 4 Error Back Propagation

## 4.2 Forward Propagation and Backward Propagation 

### 4.2.1 Forward Propagation(FP)

### 4.2.2 Backward Propagation(BP)
* Add-Node:
    * This node don't need the signal of FP. It only transports the value of the upper value to the lower value.
    ![Add](/Images/CH4_2_Add.png)
     
* Multiply-Node:
    * Signal reverse, if the signal of FP is `x`, so the signal of BP is `y`. 
    ![Mul](/Images/CH4_2_Mul.png)