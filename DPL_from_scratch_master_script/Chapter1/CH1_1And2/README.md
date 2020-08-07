

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-29 17:17:31
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-06-30 20:10:35
 * @FilePath     : /Deep_Learning/Chapter1/CH1_1And2/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
<!-- TOC -->

- [Chapter 1 Math and NumPy](#chapter-1-math-and-numpy)
    - [1.1 Data Dimension](#11-data-dimension)
    - [1.2 Data in Numpy](#12-data-in-numpy)
        - [1.2.1 Scalars](#121-scalars)
        - [1.2.2 Vectors](#122-vectors)
        - [1.2.3 Matrices](#123-matrices)
        - [1.2.4 Tensors](#124-tensors)
        - [1.2.5 Reshape](#125-reshape)
        - [1.2.6 Broadcast](#126-broadcast)
    - [Reference:](#reference)

<!-- /TOC -->
# Chapter 1 Math and NumPy

## 1.1 Data Dimension
|           | Scalar | Vector   | Matrices                | Tensor |
|-----------|--------|----------|-------------------------|--------|
|  Value    |  3.14  |  [1 2 3] | [1 2 3]<br>[4 5 6]<br>[7 8 9] |    ?   |
| Name      | Scalar | Vector   | Matrix                  | Tensor |
| Dimension | 0      | 1        | 2                       | n      |
* Index of matrices for **textbook**:  

    *   | a11 | a12 | a13 |
        |-----|-----|-----|
        | a21 | a22 | a23 |
        | a31 | a32 | a33 |
* Index of matrices for **Programmers**:

    *   | a00 | a01 | a02 |
        |-----|-----|-----|
        | a10 | a11 | a12 |
        | a20 | a21 | a22 |

## 1.2 Data in Numpy
* import **numpy**:  
`import numpy as np`  
Then you can write `np.` to use the function of **numpy**.

* Data Types:  
You can use function of numpy `ndarray`. It seems as the `list` of **python**. But it can store n-dimension to show the **data types**(scalar, vector, matrix and tensor).

### 1.2.1 Scalars

* **pass by `value`** to `np.array()`:  
Create a scalar: `s = np.array(5)`


* Python vs NumPy  
    * Python: int, float, bool and so on.  
    NumPy: not only all of *Python*, but also `uint8`, `int8`, `uint16` and so on.

* **Each item** in the array must have **the same type**. In this respect, **a NumPy array** is more like an array of `C` than a `Python` list.  

    Example:
    ```
    import numpy as np
    
    s = np.array(5)
    s.shape
    print(s.shape)
    ```
    You will get answer `()`. It means `5` is a scalar. It has 0 dimension. 
* `.shape` is a function of **numpy**. It is used to **return a [shape](https://numpy.org/doc/stable/reference/generated/numpy.shape.html?highlight=shape)** of an array. 

### 1.2.2 Vectors

* **pass by `list`** to `np.array()`:  
`v = np.array([1,2,3])`

* index:  
    ```
    >>> x = v[1]
    >>> print(x)
    2
    >>> v[1:]
    [2, 3]
    ```

### 1.2.3 Matrices

* **pass by matrix** to `np.array()`:  
`m = np.array([[1,2,3], [4,5,6], [7,8,9]])`
    ```
    >>> m.shape
    (3, 3)
    ```
* It means **`m` has 2 dimensions** and **the length of each dimension is 3**.

### 1.2.4 Tensors
* **pass by tensors** to `np.array()`:  
`t = np.array([[[[1],[2]],[[3],[4]],[[5],[6]]],[[[7],[8]],\
    [[9],[10]],[[11],[12]]],[[[13],[14]],[[15],[16]],[[17],[17]]]])`  
It means **the tensor has the dimension of 3x3x2x1**.
    ```
    >>> t.shape
    (3, 3, 2, 1)
    ```

* index:  
    ```
    >>> t[2][1][1][0]
    16
    ```

### 1.2.5 Reshape
* What is a `numpy.shape`?
    You can find the answer [here](https://numpy.org/doc/stable/reference/generated/numpy.shape.html?highlight=shape#numpy.shape). `numpy.shape` returns the elements of the shape ***tuple*** give **the lengths** of the corresponding **array dimensions**.

* You **don't change the content**, but you can reshape the data.

    Example:  
    ```
    >>> v = np.array([1,2,3,4])
    >>> v.shape
    (4,)
    ```
    It means 1 dimension. If you want to **get a matrix of 2 dimensions**. You can do it so:  
    ```
    >>> x = v.reshape(1,4)
    >>> x.shape
    (1, 4)
    ```
    Now the dimension of x is **2**. It is a **1x4 matrix**.
    * * You can also use `x = v[None, :]` to **reshape**. It is very useful for the experienced programmer. We don't explain it here.

### 1.2.6 Broadcast

* If the shape of two arrays are different, we can also sometimes use **the broadcast** of NumPy to calculate the product for these two arrays.
    * Example:  
        ```
        A = np.array([[1,2], [3,4]])
        B = np.array([10,20])
        A * B
        ```
    * We can also get the output `array([[10,40], [30,80]])`. 





## Reference:
1. Numpy:  https://numpy.org/doc/stable/reference/
