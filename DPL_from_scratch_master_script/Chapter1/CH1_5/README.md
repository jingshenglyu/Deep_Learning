

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-06-29 20:09:36
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-06-29 20:36:07
 * @FilePath     : /Deep_Learning/Chapter1/CH1_4/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter 1 Math and NumPy](#chapter-1-math-and-numpy)
    - [1.5 Matrix transpose](#15-matrix-transpose)

<!-- /TOC -->

# Chapter 1 Math and NumPy

## 1.5 Matrix transpose

* **Transposition** in Numpy with `.T` or `transpose()`:
    ```
    m = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    m.T
    ```
* Output:
    ```
    # array([[ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11],
        [ 4,  8, 12]])
    ```
* Shared data for `.T`
    ```
    m_t = m.T
    m_t[3][1] = 200
    ```
    * Output for `m_t`:
        ```
        # array([[ 1,   5, 9],
            [ 2,   6, 10],
            [ 3,   7, 11],
            [ 4, 200, 12]])
        ```
    * Output for `m`:
        ```
        # array([[ 1,  2,  3,   4],
            [ 5,  6,  7, 200],
            [ 9, 10, 11,  12]])
        ```
    * viz. the data of `m` and `m_t` are **shared**.

* If you have two matrices, but their **shape** are not same. So you can't make a multipulation for them.  *What should we do now?*
    * For example:
        ```
        inputs = np.array([[-0.27,  0.45,  0.64, 0.31]])
        weights = np.array([[0.02, 0.001, -0.03, 0.036], \
        [0.04, -0.003, 0.025, 0.009], [0.012, -0.045, 0.28, -0.067]])
        inputs.shape
        weights.shape
        ```
    * Output:
        ```
        (1, 4)
        (3, 4)
        ```
    * So `np.matmul(inputs, weights)` is wrong, because the shapes of inputs and weights are different. But we can use **`np.matmul(inputs, weights.T)`**. You can also do that `np.matmul(weights, inputs.T)`. The answer is the transposition of `np.matmul(inputs, weights.T)`.   
    * **So we can make a short conclusion, it depends on the shape that you wanted.**

* That means we should after that learn to use the **transposition** of matrices to slove the problem.
