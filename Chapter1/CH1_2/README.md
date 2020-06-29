

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-29 18:30:51
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-06-29 18:56:11
 * @FilePath     : /Deep_Learning/Chapter1/CH1_2/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
<!-- TOC -->

- [Chapter 1 Math and NumPy](#chapter-1-math-and-numpy)
    - [1.2 Matrix Operation](#12-matrix-operation)

<!-- /TOC -->
# Chapter 1 Math and NumPy

## 1.2 Matrix Operation

* We want to add `5` to a list of python.
    * python:  
        ```
        values = [1,2,3,4,5]
        for i in range(len(values)):
            values[i] += 5

        # Now values is [6,7,8,9,10]
        ```
    * NumPy:
        ```
        values = [1,2,3,4,5]
        values = np.array(values) + 5

        # Now values is a ndarray including [6,7,8,9,10].
        ``` 

* We can also use **multiply** of **matrices**. 
    ```
    x = np.multiply(some_array, 5)
    x = some_array * 5
    ```
    They do the same things(multiply of matrices).

* You can also do **the multiply of matrices**. For example: **matrix = m*n**. But for m and n, they must have the same **shape**.