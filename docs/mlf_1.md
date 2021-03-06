
# (PART) 机器学习基石 {-}

台湾大学林轩田老师开设机器学习基石和机器学习技法两门课程，难度比较大，但是内容非常好。

# 学习问题 {#problem}

## 什么是机器学习

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-15-lecture-1-the-learning-problem/ml_definition.png" width="75%" style="display: block; margin: auto;" />

这里的定义和 [Tom Mitchell, 1997 的定义类似。](https://shaocf.rbind.io/blog/chapter1-the-machine-learning-landscape/)

## 机器学习组成

以申请信用卡的问题为例，介绍机器学习问题中的常用记号以及组成成分。

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-15-lecture-1-the-learning-problem/learning_problem.png" width="75%" style="display: block; margin: auto;" />

$\mathcal{X}$：输入空间  
$\mathcal{Y}$：输出空间  
$\mathcal{D}$：训练数据，其中 $x_i, y_i (i=1, 2, ..., N)$ 分别属于空间 $\mathcal{X}, \mathcal{Y}$  
$\mathcal{H}$：假设空间，其中 $g \in \mathcal{H}$  

其中，目标函数 $f$ 是一个完全理想化的模式，而这个模式我们是几乎不可能学习到的，所以最终只能得到一个性能比较优的模式 $g$，使得 $g$ 接近 $f$。

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-15-lecture-1-the-learning-problem/practical_definition.png" width="75%" style="display: block; margin: auto;" />

$\mathcal{A}$：学习算法  
Learning model = $\mathcal{A}$ + $\mathcal{H}$，一个模型的确定需要两部分来定义，应该这样理解: 比如我们有一个监督式二分类学习问题，$\mathcal{A}$ 是我们所选择的解决这个问题的算法，比如 Logistic Regression，则 $\mathcal{H}$ 就是算法 Logistic Regression 对应的所有可能的 $g$ 的空间（即对应的参数和超参数空间），无论好的还是坏的。

## 机器学习和其它领域

本节讨论了机器学习分别和数据挖掘、人工智能、统计学的关系，每个学科都有其相近的地方，但又都有其不同的方向。

### 机器学习和数据挖掘

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-15-lecture-1-the-learning-problem/ml_dm.png" width="75%" style="display: block; margin: auto;" />

### 机器学习和人工智能

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-15-lecture-1-the-learning-problem/ml_ai.png" width="75%" style="display: block; margin: auto;" />

### 机器学习和统计

<img src="https://raw.githubusercontent.com/shaocf/picturebed/master/2019-11-15-lecture-1-the-learning-problem/ml_s.png" width="75%" style="display: block; margin: auto;" />






