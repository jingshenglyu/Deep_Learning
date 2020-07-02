

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-02 21:55:39
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-02 23:22:06
 * @FilePath     : /Deep_Learning/Chapter3/CH3_4/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
# Chapter 3 Data Learning in Neural Network

## 3.4 Numerical Differentiation
### 3.4.1 Derivative:  
* Error:
    * There is an **error** between f(x+h) and f(x)
    * In order to decrease this error, we will calculate the difference **between f(x+h) and f(x-h)**.
    * Diagramm for this error
        
        ![Difference](/Images/CH3_4_De.png)

### 3.4.2 Partial Derivative
* f(x0, x1) = x0 ** 2 + x1 ** 2  
    * the derivative for every variable(x0 or x1)
### 3.4.3 Gradient
* the **vector** from the partial derivatives