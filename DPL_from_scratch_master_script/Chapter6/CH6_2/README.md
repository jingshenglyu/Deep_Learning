

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-07 07:35:14
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-07 18:20:17
 * @FilePath     : /Deep_Learning/Chapter6/CH6_2/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

<!-- TOC -->

- [Chapter6 Convolution Neural Network](#chapter6-convolution-neural-network)
    - [6.2 CNN Layer](#62-cnn-layer)
        - [6.2.1 Fully Connection vs Local Connection](#621-fully-connection-vs-local-connection)
        - [6.2.2 Feature Map](#622-feature-map)
        - [6.2.3 Convolution/Filter Operation](#623-convolutionfilter-operation)
        - [6.2.4 Weights and Bias for CNN](#624-weights-and-bias-for-cnn)
        - [6.2.5 Padding and Stride](#625-padding-and-stride)
        - [6.2.6 n-dim for CNN](#626-n-dim-for-cnn)

<!-- /TOC -->

# Chapter6 Convolution Neural Network

## 6.2 CNN Layer
### 6.2.1 Fully Connection vs Local Connection
* Fully:  
The **shape** of the data is **ignored**. The Information of shape isn't used by us.
* Local:  
Keeping the data in shape  
Example: 3-dim Input Layer --> CNN Layer --> 3-dim Output Layer

### 6.2.2 Feature Map
* Input data of CNN Layer:  
**Input Feature Map**
* Output data of CNN Layer:  
**Output Feature Map**

### 6.2.3 Convolution/Filter Operation
* It is also called **filter operation**.
* Filter:  
    ![Conv](/Images/CH6_2_1_Conv.png)
    * Calculation:  
    **Multiply** the elements of the filter at each position and the corresponding elements of the input, then **sum** them up and save the result to the corresponding position of the output.

### 6.2.4 Weights and Bias for CNN
* ![WB](/Images/CH6_2_2_WB.png)
    * Filter <--> Weights
    * Constant Values <--> Bias

### 6.2.5 Padding and Stride
* Padding:  
    **Change the shape** (***larger***) of the output data
    ![PD](/Images/CH6_2_3_PD.png)

* Stride:  
    **Change the shape** (***smaller***) of the output data
    ![Stride](/Images/CH6_2_4_Stride.png)

* Example:  
    * Shape of input data = (H, W)  
        Shape of filters = (FH, FW)  
        Shape of output data = (OH, OW)  
        Padding = P  
        Stride = S  
    
* Result:
    * OH = ((H + 2P - FH) / S) +1  
        OW = ((W + 2P - FW) / S) +1

### 6.2.6 n-dim for CNN
* expression:   
    1. for 3-dim (channel, height, width)  
    2. channel of input = channel of filter  
    3. Every channel has **only one bias**.

* Diagramm of 1-dim output data for CNN
![1D_OP](/Images/CH6_2_5_1D_OUTPUT.png)

* Diagramm of n-dim output data for CNN
![ND_OP](/Images/CH6_2_6_ND_OUTPUT.png)

* Diagramm of n-dim output data for CNN plus bias
![P_BIAS](/Images/CH6_2_7_PLUS_BIAS.png)
