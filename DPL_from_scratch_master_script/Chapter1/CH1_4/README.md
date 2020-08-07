

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-29 19:35:07
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-06-29 21:43:49
 * @FilePath     : /Deep_Learning/Chapter1/CH1_4/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter 1 Math and NumPy](#chapter-1-math-and-numpy)
    - [1.4 Multiplying the matrices](#14-multiplying-the-matrices)
        - [1.4.1 Multiplying in NumPy:](#141-multiplying-in-numpy)
        - [1.4.2 Dot Product](#142-dot-product)

<!-- /TOC -->
# Chapter 1 Math and NumPy

## 1.4 Multiplying the matrices

* The number of **columns** in the left matrix must **equal** the number of **rows** in the right matrix.
* The *end matrix* always has the same number of **rows** as the matrix on the **left** and the same number of **columns** as the matrix on the **right**.
* The order is important: multiplication **A•B is not equal** to multiplication **B•A** .
* The data in the left-hand matrix should be arranged as rows, while the data in the right-hand matrix should be arranged as columns.

### 1.4.1 Multiplying in NumPy:  
* Multiply
    * Scalar multiply matrix  
        ```
        m = np.array([[1,2,3],[4,5,6]])
        n = m * 0.25
        n
        ```
    * Output:  
        ```
        array([[ 0.25,  0.5 ,  0.75],
            [ 1.  ,  1.25,  1.5 ]])
        ```
    * Matrix multiply matrix
        ```
        m * n
        np.multiply(m, n)
        ```
        They are same. Output:
        ```
        array([[ 0.25,  1.  ,  2.25],
            [ 4.  ,  6.25,  9.  ]])
        ```

### 1.4.2 Dot Product

* For 2-dim array they(`matmul` and `dot`) are same. But for n-dim, they are not always same. 
    * `matmul`
        ```
        a = np.array([[1,2],[3,4]])
        np.dot(a,a)
        ```
    * Output: 
        ```
        array([[ 7, 10],
            [15, 22]])
        ```
    * `dot`
        ```
        np.dot(a,a)
        ```
    * Output: 
        ```
        array([[ 7, 10],
            [15, 22]])
        ```
* You can check the difference bbetween [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html#numpy.matmul) and [dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html).
* [Difference](https://blog.csdn.net/yexiaohhjk/article/details/82659818) in Chinese 


