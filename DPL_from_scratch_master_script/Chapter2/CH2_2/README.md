

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-30 20:28:50
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-06-30 20:35:03
 * @FilePath     : /Deep_Learning/Chapter2/CH2_2/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter 2 Introduction to Neural Networks](#chapter-2-introduction-to-neural-networks)
    - [2.2 Logical Connective](#22-logical-connective)

<!-- /TOC -->
# Chapter 2 Introduction to Neural Networks

## 2.2 Logical Connective
* And-gate:  
    ```
    def AND(x1, x2):
        w1, w2, theta = 0.5, 0.5, 0.7
        tmp = x1*w1 + x2*w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
        return 1
    ```

* Or-gate

* Not-gate

* XOR-gate:  
This is the limit for **perceptron**. Perceptron can only classify **the linear parts**.