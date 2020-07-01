

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-01 17:11:08
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-01 17:56:08
 * @FilePath     : /Deep_Learning/Chapter2/CH2_5/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
# Chapter 2 Introduction to Neural Networks

## 2.5 Neural Networks 
### 2.5.1 Dot Product
* Example:      
    ```
    X = np.array([1, 2])
    W = np.array([1, 3, 5], [2, 4, 6])

    Y = np.dot(X, W)
    print(Y)
    ```
* Output:  
    ```
    [5 11 17]
    ```
* ! np.dot(X, W) is wrong because of the shape.

### 2.5.2 Three Layers Neural Networks
* Diagramm for 3-layers NN
    ![3-layers NN](/Images/CH2_5_2_3LayerNN.png)
* We want to get output `y1 and y2` of the output-layer. 
[Code is here]()

* Calculate the 3-layers NN with diagramms
    ![0](/Images/CH2_5_2_3LayerNN_0thLayer.png)
    ![1](/Images/CH2_5_2_3LayerNN_1stLayer_af.png)
    ![2](/Images/CH2_5_2_3LayerNN_1stLayer_2ndLayeraf.png)
    ![3](/Images/CH2_5_2_3LayerNN_2ndLayer.png)
