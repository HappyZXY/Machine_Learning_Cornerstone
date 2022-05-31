## 西瓜书阅读笔记之SVM

### 一、SVM的基本型

首先了解一下SVM是干什么的，SVM用来分类样本的。SVM的目标是寻找到一个最佳的超平面使得（超平面可能有很多，最佳超平面和支持向量之间的间隔最可能大）。划分超平面可以通过线性方程来描述：
$$
w^T\boldsymbol{x}+b = 0
$$

$ w=(w_1;w_2;...;w_d) $为法向量，决定了超平面的方向，$b$为位移项，据定了超平面与原点之间的距离，$w,b$是确定超平面的两个唯一的重要因素，记为$(w,b)$，样本空间中任意点$x$到超平面$(w,b)$的距离可写为：
$$
r = \frac{|w^T\boldsymbol{x}+b|}{||w||}
$$
假设超平面$(w,b)$能够训练样本正确分类，即对于$(x_i, y_i) \in D$，若$y_i = +1$，则有$w^Tx_i+b>0$；若$y_i=-1$，则有$w^Tx_i+b<0$，根据切比雪夫不等于，我们一定可以找到一个$\epsilon$ 满足$w^Tx_i+b≤\epsilon<0$,$\epsilon＜0$ 
$$
w^T\boldsymbol{x}_i+b≤ \epsilon < 0
$$

$$
w^T\boldsymbol{x}_i+b≤-1， y_i=-1
$$

$$
两边同时除以-\epsilon, \frac{-w^Tx_i}{-\epsilon}-\frac{b}{\epsilon}≤-1
$$

$$
同理，w^T\boldsymbol{x}_i+b≥+1， y_i=+1
$$

$$
\
\left\{\begin{array}{lr}   
w^T\boldsymbol{x}_i+b≤-1, \quad y_i=-1 \\ 
w^T\boldsymbol{x}_i+b≥+1, \quad y_i=+1   \quad\quad\quad\quad\quad  (6.3)
\end{array}\right.
$$



距离超平面最近的这几个样本点使得（1）（2）成立，它们被称为“支持向量”(support vector)，图6.2中表示为类似于正负电荷的圆圈，两个异类支持向量到超平面的距离之和为$\gamma$，$\gamma$也被称为“间隔”(margin)
$$
\gamma = \frac{2}{||w||}=\frac{1}{||w||}+\frac{1}{||w||}
$$


![](https://img2022.cnblogs.com/blog/1195588/202204/1195588-20220420091903333-1827856908.png)

欲找到具有“最大间隔（maximum margin）”的划分超平面，也就是要找到能满足约束条件（1）（2）的参数$w$和$b$，使得$\gamma$最大，即
$$
\max _{w, b} \frac{2}{\|\mathcal{w}\|}    \quad\quad\quad\quad(6.5)
$$

$$
s.t. \quad y_i(w^T\boldsymbol{x}_i+b)≥1, i=1,2,...,m
$$

显然，为了最大化间隔，仅需**最大化**$\frac{1}{||w||^2}=(||w||^2)^-1$，这等价于$||w||^2$最小化，于是上式可以被重写为：
$$
\min _{w, b} \frac{1}{2}{\|\mathcal{w}\|}^2 =f(w)  \quad\quad\quad\quad\\
s.t. \quad y_i(w^T\boldsymbol{x}_i+b)≥1, i=1,2,...,m  \quad \quad\quad\quad (6.6)
$$
这就是支持向量机（Support Vector Machine，简称SVM）的基本型。



### 二、对偶问题

目标和约束条件都有了，现在就是要解式（6.6）来得到最大间隔划分超平面所对应的模型

当我们要求一个函数 min f(x) 的时候，如果 f(x) 可导，我们通过是通过求 f(x) 的导数来得。

但是如果函数 f(x) 带约束条件，如（6.6），那么问题就开始变复杂了。**凸优化的目标就是解决带约束条件函数的极值问题[https://zhuanlan.zhihu.com/p/89292221?from_voters_page=true]**。

凸优化解决的通用模型是：
$$
\begin{cases}\min & f(x) \\ \text { s.t. } & g_{i}(x) \leq 0, \quad i=1, \ldots, m \\ & h_{i}(x)=0, \quad i=1, \ldots, n\end{cases}
$$
不是所有的极值问题都可以适用的**凸优化理论，它必须满足以下条件**：

1、目标函数 f(x) 为凸函数 (二阶可导，且二阶导数＞0)

2、不等式约束函数 g(x) 为凸函数

3、等式约束函数 h(x) 为仿射函数 (仿射函数=导数最高阶数为1）

只有同时满足以上3个条件，才属于凸优化的范畴。

<font size=1.5>凸函数：定义域为凸集，凸集几何意义表示为：如果集合中任意2个元素连线上的点也在集合C中，则C为凸集 </font>

在(6.6)中，$w$和$b$是模型参数，目标函数$\min_{w, b} \frac{1}{2}{\|\mathcal{w}\|}^2$是二次函数，凸函数。约束条件$s.t. y_i(w^Tx_i+b)≥1$是仿射函数，所以SVM本身十一个凸二次规划（convex quadratic programming）问题
$$
g(w, b)=1-y_{i}\left(w^{T} \boldsymbol{x}_{i}+b\right)
$$
<font color='darkblue'>对于凸优化的通用模型，由于其带有约束条件，很难处理，因此我们会考虑怎么用一个式子来表述那个通用模型呢？拉格朗日乘子法就是一个很好的方法，使用拉格朗日乘子法可以得到其"对偶问题"(dual problem)，具体来说，对式(6.6)的每个约束条件添加拉格朗日乘子$\alpha_i≥0$（如果$\alpha_i$与$g(w,b)$同号，最大值就是∞，没有意义），则该问题的拉格朗日函数可以些写为：</font>
$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{i}\left(w^{T} \boldsymbol{x}_{i}+b\right)\right) \quad
\quad \quad \quad (6.8)
$$
上述问题的等价问题为：
$$
\min -\max L(w, b, \alpha)
$$
其对偶问题为：
$$
\max -\min L(w, b, \alpha)
$$
先求**min(L)**，$\alpha=\alpha_1;\alpha_2;...;\alpha_m$。令$L(w, b, \alpha)$对$w$和$b$的偏导为零可得
$$
\boldsymbol{w}=\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i} \quad \quad \quad (6.9)
$$

$$
0=\sum_{i=1}^{m} \alpha_{i} y_{i} \quad \quad \quad \quad \quad (6.10)
$$

将(6.9)代入(6.8):
$$
=\frac{1}{2}\sum_{i=1}^{m} (\alpha_{i} y_{i} \boldsymbol{x}_{i})^2+\sum_{i=1}^{m} \alpha_{i}\left(1-y_{j}\left(\sum_{j=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}^T \boldsymbol{x}_{j}+b\right)\right)\\

=\frac{1}{2}\sum_{i=1}^{m} (\alpha_{i} y_{i} \boldsymbol{x}_{i})^2+\sum_{i=1}^{m} \alpha_{i}-\alpha_i y_j\sum_{j=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}^T \boldsymbol{x}_{j}+b \\

\sum_{i=1}^{m}\alpha_i-
\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}-\alpha_iy_j\sum_{j=1}^mb  \quad \quad 代入6.10 \\ 
\sum_{i=1}^{m}\alpha_i-
\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j}
$$
于是
$$
\
\left\{\begin{array}{lr} 
\max_\alpha\sum_{i=1}^{m}\alpha_i-
\frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}_{j} \\
s.t. \quad 0=\sum_{i=1}^{m} \alpha_{i} y_{i},\quad \alpha≥0，i=1,2,...,m
\end{array}\right. \quad \quad \quad \quad(6.11)
$$
解出$\alpha$后，求出$w$ 和$b$即可得到模型
$$
\begin{aligned}
f(\boldsymbol{x}) &=\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b \\
&=\sum_{i=1}^{m} \alpha_{i} y_{i} \boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}+b .
\end{aligned} \quad \quad \quad \quad \quad (6.12)
$$
从对偶问题(6.11)解出的$\alpha_i$是式(6.8)中的拉格朗日乘子，它恰对应训练样本$(\boldsymbol{x}_i,y_i)$。如果能满足KKT条件，**原始问题=对偶问题**。
$$
\left\{\begin{array}{l}
\alpha_{i} \geqslant 0 ; \\
y_{i} f\left(\boldsymbol{x}_{i}\right)-1 \geqslant 0 ; \\
\alpha_{i}\left(y_{i} f\left(\boldsymbol{x}_{i}\right)-1\right)=0
\end{array}\right.
$$
<font color='darkblue'>模型最终转化成参数为$\alpha$的函数，求最大值问题，也就是有$w$,$b$两个参数变成一个参数</font>

于是，对于任意训练样本$(\boldsymbol{x}_i,y_i)$，总$\alpha_i =0 $有或$y_if(\boldsymbol{x_i}=1)$。若，则该样本将不会在（6.12）的求和中出现，也就不会对$f(\boldsymbol{x})$有任何影响；若$\alpha_i$>0，则必有$y_if(\boldsymbol{x_i}=1)$，所对应的样本点位于最大间隔边界上，是一个支持向量。这显示出支持向量机的一个重要特性：训练完成后，大部分的训练样本都不需要保留，最终模型仅与支持向量有关。

那么现在的问题重点又转到求解式子（6.11），不难发现这是一个二次规划问题，可使用通过的二次规划算法来求解，该问题的规模正比于训练样本数，这会在实际任务中造成很大的开销，为了避免这个问题，人们通过利用问题本身的特性，提出了很多高效的算法，SMO(Sequential Minimal Optimization)是一个代表，SMO的思想是先固定$\alpha_i$之外的所有参数，然后求$\alpha_i$上的极值。



### 三、核函数

在前面的公式推理前提是假设样本线性可分，即存在一个划分超平面能将训练样本正确分类，然而在现实生活中，原始样本空间也许并不存在，对于这样的问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。

我们希望样本在特征空间内线性可分，因此特征空间的好坏对支持向量机的性能至关重要。需要注意的是，在不知道特征映射的形式时，我们并不知道什么样的核函数最合适，而核函数也仅是隐式地定义了这个特征空间。于是，“核函数选择”成为支持向量机的最大变数。若核函数选择不佳，很可能导致性能不佳！！

![hehanshu](Machine_Learning_Cornerstone/SVM/pic/hehanshu.png)



### 四、软间隔与正则化

现实生活中往往很难确定合适的核函数是的训练样本在特征空间中线性可分；退一步说，即便恰好找到了某个核函数使训练样本在特征空间中线性可分，也很难推断这个貌似线性可分的结果不是由于过拟合所造成的。

缓解这个问题的一个办法是**允许支持向量机在一些样本上的出错**。为此，要引入“软间隔”（soft margin）的概念，如下图6.4所示：

![ruanjiange](Machine_Learning_Cornerstone\SVM\pic\ruanjiange.png)

前面介绍的支持向量机形式时保证所有样本正确划分，这称为“硬间隔”（hard margin），而软间隔是允许某些样本不满足条件：
$$
y_{i}\left(w^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1 \quad \quad \quad (6.28)
$$
当然，在最大化间隔的同时，不满足约束的样本要尽可能少，于是，优化目标可写为：
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right) \quad \quad \quad (6.29)
$$
其中$C>0$是一个常数，$\ell_{0 / 1}$是“0/1损失函数”
$$
\ell_{0 / 1}(z)= \begin{cases}1, & \text { if } z<0 \\ 0, & \text { otherwise. }\end{cases} \quad \quad \quad \quad (6.30)
$$
显然，当$C$无穷大时，式（6.29）迫使所有样本满足约束条件（6.28），于是式（6.29）等价于（6.6）；当$C$取有限值时，式（6.29）允许一些样本不满足约束。

然而，$\ell_{0 / 1}$非凸、非连续，数学性质不太好，使得式（6.29）不易直接求解，于是人们通常用其他的一些函数来代替$\ell_{0 / 1}$，称为“替代损失”（surrogate loss）。替代损失函数一般具有较好的数学性质，如它们通常是凸函数且是$\ell_{0 / 1}$的上界。图6.5给出了三种常用的替代损失函数：

hinge 损失: $\ell_{\text {hinge }}(z)=\max (0,1-z)$;
指数损失(exponential loss): $\ell_{\exp }(z)=\exp (-z)$;
对率损失(logistic loss $): \ell_{\log }(z)=\log (1+\exp (-z))$.

若采用hinge损失，式（6.29）变成
$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right) \quad \quad \quad (6.34)
$$
引入“松弛变量”（slack variables）$\xi_{i} \geqslant 0$，可将式（6.34）重写为：
$$
\min _{w, b, \xi_{i}} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i}  \quad \quad \quad (6.35)
$$

$$
\begin{array}{ll}
\text { s.t. } & y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1-\xi_{i} \\
& \xi_{i} \geqslant 0, i=1,2, \ldots, m .
\end{array}
$$

这就是常用的“软间隔支持向量机”。

![tidai](Machine_Learning_Cornerstone\SVM\pic\tidai.png)

显然，式（6.35）中每个样本都有一个对应的松弛变量，用以表征该样本不满足约束（6.28）的程度。但是，与式（6.6）相似这仍然是一个二次规划问题。于是，类似（6.8），通过拉格朗日乘子法可得到式（6.35）的拉格朗日函数：
$$
\begin{aligned}
L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu})=& \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \xi_{i} \\
&+\sum_{i=1}^{m} \alpha_{i}\left(1-\xi_{i}-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)-\sum_{i=1}^{m} \mu_{i} \xi_{i}
\end{aligned} \quad \quad \quad (6.36)
$$
其中，$\alpha_{i} \geqslant 0, \mu_{i} \geqslant 0$是拉格朗日乘子。其后和上述流程一样，对参数求导，求拉格朗日乘子，KKT条件
