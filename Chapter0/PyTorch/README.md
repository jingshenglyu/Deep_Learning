

<!--
 * @Author       : Jingsheng Lyu
 * @Date         : 2020-07-08 17:52:24
 * @LastEditors  : Jingsheng Lyu
 * @LastEditTime : 2020-07-09 22:39:59
 * @FilePath     : /Deep_Learning/Chapter0/PyTorch/README.md
 * @Github       : https://github.com/jingshenglyu
 * @Web          : https://jingshenglyu.github.io/
 * @E-Mail       : jingshenglyu@gmail.com
--> 

# Deep Learning with PyTorch

* In this project, I learn Deep Learning with this book-[Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf). It has also a [Chinese website](https://tangshusen.me/Deep-Learning-with-PyTorch-Chinese/#/) for the translation. But it's shorter as the original book. 

# Install PyTorch

* PC: Ubuntu 18.04 LTS
* GPU: None
* IDE: Anaconda & PyCharm

## 1. Using PyCharm to install PyTorch(Recommend)
1. Using [PyCharm website](https://www.jetbrains.com/pycharm/download/#section=linux) to install PyCharm (Community is for free) 

2. Create a new project by PyCharm

3. Top left： File -> Settings; Then choose Project:Your Project Name -> project interpreter: Python 3.7 by Conda -> choose "+" to install **PyTorch**
![Project Interpreter](/Images/ProjectInterpreter.png)

4. input the followed code to make a verification for PyTorch:  
* Code:
    ```
    import torch
    print(torch.__version__)

    x = torch.randn(2,3)
    print(x)
    ```
* Output:
    ```
    1.5.0
    tensor([[-1.2200,  0.2054,  0.9362],
            [ 0.4059, -0.1795, -0.5834]])
    ```

## 2. Install PyTorch by terminal
1. Visit [PyTorch](https://pytorch.org/)
2. choose your OS, package, language and so on...
3. Copy the code after `Run this Command:` in terminal