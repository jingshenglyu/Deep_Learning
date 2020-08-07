

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-07 07:35:16
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-08 17:51:27
 * @FilePath     : /Deep_Learning/Chapter6/CH6_3/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter6 Convolution Neural Network](#chapter6-convolution-neural-network)
    - [6.3 Pooling Layer](#63-pooling-layer)
        - [6.3.1 Purpose:](#631-purpose)
        - [6.3.2 Characters of Pooling Layer](#632-characters-of-pooling-layer)
        - [6.3.3 Base on im2col](#633-base-on-im2col)

<!-- /TOC -->
# Chapter6 Convolution Neural Network

## 6.3 Pooling Layer
### 6.3.1 Purpose:
* Get **the special value** from this process      
    For example: the maximal value or the average and so on

### 6.3.2 Characters of Pooling Layer
1. no parameters for learning. Because it will only get the value 
2. don't change the number of the channels

### 6.3.3 Base on im2col
* function `im2col`:  
    It is called **"image to column"**
    data of 4-dim --> array of 2-dim
    ![im2col](/Images/CH6_3_1_im2col.png)

* im2col characters:  
    1. the number of im2col parameters will be **larger** as the number of the original parameters

    2. Consume more memory


