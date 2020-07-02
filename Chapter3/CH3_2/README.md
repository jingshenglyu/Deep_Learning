

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-02 21:55:35
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-02 22:59:51
 * @FilePath     : /Deep_Learning/Chapter3/CH3_2/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 
# Chapter 3 Data Learning in Neural Network

## 3.2 Loss Function
### 3.2.1 **Generalization** ability:  
* An ability that can process the **untraining** data. (泛化能力)

    * training data: supervised data
    * test data

### 3.2.2 Over fitting:  
* Process **only one** dataset, **can't** process the other datasets.

### 3.2.3 Loss(Cost) Function
* Using loss function **to find the best weights**
    * Loss function:  
        * mean squared error
            * 0.5 * np.sum(y_k - t_k ** 2)
        * cross entropy error    
        *   ```
            def cross_entropy_error(y,t):
            delta = 1e-7
                return -np.sum(t * np.ln(y + delta))
            ```

* Why loss function?
    * To find the weights that makes the value of the *loss function* as *small* as possible. In the place where the weight is located, you need to calculate the **derivative** of the weights (the ***gradient*** to be exact) and use this derivative as a guide. The value of the weight is updated gradually.

