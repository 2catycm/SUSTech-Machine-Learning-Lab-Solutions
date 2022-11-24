# Lab10.SVM
叶璨铭

## Exercise 1
调参motivation
- 视频中的车辆比较小，所以窗口大小应该比较小
- hog feature 不太好用，实验发现spatial比较好
- 右下角的车辆笔记多，划窗范围可以减少。

### Exercise 2 Questions(4 points)

1. Can SVM be used for unsupervised clustering or data dimension reduction? Why?
SVM训练后，保留了少量的支持向量用于决策，发现了数据中真正重要的样本。
但是这个不叫降维，降维是指将数据从高维降到低维，降低特征量。
SVM一定程度上可以解决维数灾难，in that SVM避免了高维情况下需要很多样本去决策，参数量过大的问题。
但是并不能说SVM是降维，更不是聚类。

2. What are the strengths of SVMs; when do they perform well?
当SVM的数据集线性可分时，SVM的表现很好。
如果数据集线性不可分，就得看运气，就是你kernel函数选择的好不好，如果选择的好，
映射后的高维数据集线性可分，SVM效果也好，不然还是线性不可分，那就不好了。

3. What are the weaknesses of SVMs; when do they perform poorly?
首先，数据量太大的时候，SVM训练比较慢，因为SVM训练是个优化问题，需要很多次迭代。
其次，如2.中所说的，万一核函数选的不好，而且软边距也不行，那么SVM就不好用了。

4. What makes SVMs a good candidate for the classification / regression problem, if you have enough knowledge about the data?
如果我知道数据是线性可分的，那么SVM是个不错的选择，这个时候是极大边距的，泛化性能某种意义来讲是最好的。