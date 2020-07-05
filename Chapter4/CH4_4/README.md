

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-04 08:42:03
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-05 22:37:34
 * @FilePath     : /Deep_Learning/Chapter4/CH4_4/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter 4 Error Back Propagation](#chapter-4-error-back-propagation)
    - [4.4 Activation Function for Computational Graph of Neural Network](#44-activation-function-for-computational-graph-of-neural-network)
        - [4.4.1 Activation Function](#441-activation-function)
        - [4.4.2 Affine Layer](#442-affine-layer)

<!-- /TOC -->

# Chapter 4 Error Back Propagation

## 4.4 Activation Function for Computational Graph of Neural Network
### 4.4.1 Activation Function
* ReLU function:  
    ![ReLU](/Images/CH4_3_ReLU.png)
* Sigmoid:  
    ![Sigmoid](/Images/CH4_3_Sigmoid.png)

### 4.4.2 Affine Layer
* The **multiplication of matrices** in the *forward* propagation of neural networks is known in the field of geometry as the "affine" operation. The ***"Affine Transform"***. Therefore, the processing for affine transformation is implemented here as ***"Affine layer"***.
    ![Affine](/Images/CH4_3_Affine.png)