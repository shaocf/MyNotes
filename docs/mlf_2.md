
# 学习回答 Yes/No {#yes_or_no}

## 感知假设集（Perceptron Hypothesis Set）

这里还是以是否授予用户信用卡为例，引出一个简化的假设集 $\mathcal{H}$: Perceptron（感知机）。



<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/perceptron.png" width="75%" style="display: block; margin: auto;" />

感知机假设可以写成向量的形式：

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/perceptron_vector.png" width="75%" style="display: block; margin: auto;" />

那么感知机 $h$ 会是什么样的呢，可以通过在二维空间构建一个感知机。

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/perceptron_r2.png" width="75%" style="display: block; margin: auto;" />

感知机在二维空间中是一个线性的二分类器，在多维空间中是分类超平面。

## 感知学习算法（Perceptron Learning Algorithm）

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA_H.png" width="75%" style="display: block; margin: auto;" />

空间 $\mathcal{H}$ 表示所有可能的感知机，我们如何选择一个 $g$ ？  

* 我们想得到 $g \approx f$，但由于我们无法获知 $f$，所以是很难的

* 但是我们可以得到在 $\mathcal{D}$ 上 $g \approx f$，更理想的状况下 $g(x_n) = f(x_n) = y_n$  

* 难点是 $\mathcal{H}$ 是个无穷集合

* 方法：从某个 $g_0$ 开始，修正它在 $\mathcal{D}$ 上犯的错误

**下面利用权重向量 $\mathrm{w}_0$ 来表示某个感知机 $g_0$**

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA.png" width="75%" style="display: block; margin: auto;" />

$\mathrm{w_t}$ 表示在第 $t$ 轮，感知机 $\mathrm{w_t}$ 在点 $(x_{n(t)}, y_{n(t)})$ 犯错误，有两类错误：

1. 如果 $y_{n(t)}$ 为正，而预测为负，即 $\mathrm{w_t}^Tx_{n(t)} < 0$，则向量 $\mathrm{w_t}$ 和向量 $x_{n(t)}$ 符号之间夹角为钝角，则通过 $\mathrm{w_{t+1}} = \mathrm{w_t} + y_nx_{n(t)}$ 使得和 $x_{n(t)}$ 夹角更小一些；

2. 如果 $y_{n(t)}$ 为负，而预测为正，即 $\mathrm{w_t}^Tx_{n(t)} > 0$，则向量 $\mathrm{w_t}$ 和向量 $x_{n(t)}$ 符号之间夹角为锐角，则通过 $\mathrm{w_{t+1}} = \mathrm{w_t} + y_nx_{n(t)}$ 使得和 $x_{n(t)}$ 夹角更大一些；

直到不会犯错误为止，最后得到的 $\mathrm{w}$ 称之为 $\mathrm{w_{PLA}}$。

下面是一道选择题，感觉理解起来还是有点意思的，因此在这里也记录一下。

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA_work.png" width="75%" style="display: block; margin: auto;" />

由于 $y_n\mathrm{w_t}^Tx_n < 0$，而 $y_n\mathrm{w_{t+1}}^Tx_n >= y_n\mathrm{w_t}^Tx_n$，说明如果 $y_n\mathrm{w_{t+1}}^Tx_n >= 0$ 则 $\mathrm{w_{t+1}}$ 对 $x_n$ 不再犯错，即使仍然犯错，它也比 $\mathrm{w_t}$ 使得 $y_n\mathrm{w_{t+1}}^Tx_n$ 更接近 0，仍然是作了一部分修正。

## PLA 的确定性（Guarantee of PLA）

**PLA 能找到最终不犯错误的 $\mathrm{w}$，即 PLA 停止的充分必要条件是：$\mathcal{D}$ 线性可分。**

1. 首先，如果 PLA 停止，也就是存在某个 $\mathrm{w}$，使得在 $\mathcal{D}$ 上不犯错误，则称 $\mathcal{D}$ 是线性可分的。

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/linear_separablility.png" width="75%" style="display: block; margin: auto;" />

2. 那么反过来，假设 $\mathcal{D}$ 线性可分，那么 PLA 会停止吗？下面来证明：

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA_align.png" width="75%" style="display: block; margin: auto;" />

$\mathrm{w_t}$ Gets More Aligned with $\mathrm{w_f}$ 可解释为：由于 $\mathrm{w_f}^T\mathrm{w_{t+1}} > \mathrm{w_f}^T\mathrm{w_t}$，所以 $\mathrm{w_{t+1}}$ 比 $\mathrm{w_t}$ 更接近 $\mathrm{w_f}$（可以理解为更近乎相等），即夹角更小；这里有个问题，为什么这种变化不是由于 $\mathrm{w_{t+1}}$ 比 $\mathrm{w_t}$ 的模（向量长度）更大引起的呢？下面一幅图给出解释，即通过公式 $||\mathrm{w_{t+1}}||^2 \le ||\mathrm{w_t}||^2 + \max_{n}||y_nx_n||^2$ 来看，$\mathrm{w_{t+1}}$ 的长度是有范围的，不会增长的太快。

在上面图片的式子中，可进一步得到

\begin{align}
\mathrm{w_f}^T\mathrm{w_{t+1}} & = \mathrm{w_f}^T(\mathrm{w_t} + y_{n(t)}x_{n(t)}) \\
& \ge \mathrm{w_f}^T\mathrm{w_t} + \min{n}y_n\mathrm{w_f}^Tx_n \\
& \ge \mathrm{w_f}^T\mathrm{w_0} + (t+1)\min_{n}y_n\mathrm{w_f}^Tx_n
\end{align}

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA_fact.png" width="75%" style="display: block; margin: auto;" />

\begin{align}
||\mathrm{w_{t+1}}||^2 & \le ||\mathrm{w_t}||^2 + \max_{n}||y_nx_n||^2 \\
& \le ||\mathrm{w_0}||^2 + (t+1)\max_{n}||y_nx_n||^2
\end{align}

由上面计算的两个式子可以得到：

定义：

$$R^2 = \max_{n}||x_n||^2$$  
$$\rho = \min_{n}y_n\frac{\mathrm{w_f}^T}{||\mathrm{w_f}||}x_n$$

则从 $\mathrm{w_0} = 0$ 开始，经过 $T$ 轮错误纠正可以得到：

\begin{align}
\frac{\mathrm{w_f}^T}{||\mathrm{w_f}||}\frac{\mathrm{w_T}}{||\mathrm{w_T}||} & \ge \frac{T\min_{n}y_n\mathrm{w_f}^Tx_n}{||\mathrm{w_f}||\sqrt{T}\max_{n}||y_nx_n||} \\
& \ge \frac{\sqrt{T}\rho}{R} \\
& = \sqrt{T}\cdot constant 
\end{align}

由上面得到的公式，可以直接得到下面练习题的答案：

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA_fun.png" width="75%" style="display: block; margin: auto;" />

## 非可分数据集（Non-Separable Data）

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/PLA_more.png" width="75%" style="display: block; margin: auto;" />

* PLA 的优点：实施起来比较简单  
* 缺点是“假设” $\mathcal{D}$ 是线性可分的，但是，是否线性可分我们并不知道；即使我们知道其线性可分，至于它多久会停下来，我们也不知道。

因此我们需要解决带有噪音的数据上的学习过程，即 $\mathcal{D}$ 不是线性可分的，因此引入了 Pocket 算法。

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/noisy_data.png" width="75%" style="display: block; margin: auto;" />

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/line_noise.png" width="75%" style="display: block; margin: auto;" />

以上求解 $\mathrm{w_g}$ 的问题是 NP-hard 问题。

因此，我们可以退而求其次，使用贪心算法 Pocket 算法来求解：

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-17-lecture-2-learning-to-answer-yesno/pocket.png" width="75%" style="display: block; margin: auto;" />






