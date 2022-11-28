# Machine Learning Homework 4: Linear Models for Classification

> 叶璨铭， 12011404@mail.sustech.edu.cn


## Discriminant Function: Maximum Class Separation

> ![image-20221128111518172](P_Homework4_叶璨铭.assets/image-20221128111518172.png)

$$
\mathop{\mathbf{max}}_{w} f(w) = w^T{(\mathbf{m_2}-\mathbf{m_1})}\\
s.t. w^Tw = 1
$$

Using Lagrange Multiplier $\lambda$, we can transform the problem to be unconstrained. 
$$
\nabla f(w) = \lambda \nabla g(w) \\
g(w) = w^Tw-1 \\
g(w) = 0
$$
Since $\nabla f(w) = (\mathbf{m_2}-\mathbf{m_1})^T$ and $\nabla g(w) = 2w^T$, we obtain 
$$
w = \frac{1}{2\lambda}(\mathbf{m_2}-\mathbf{m_1}) \sim (\mathbf{m_2}-\mathbf{m_1})
$$


## Discriminant Function: Fisher Criterion

> ![image-20221128172805560](P_Homework4_叶璨铭.assets/image-20221128172805560.png)

## Generative Classification Model

> ![image-20221128172850130](P_Homework4_叶璨铭.assets/image-20221128172850130.png)

## Discriminative Classification Model

> ![image-20221128173001671](P_Homework4_叶璨铭.assets/image-20221128173001671.png)



## Discriminative Classification Model

> ![image-20221128173038275](P_Homework4_叶璨铭.assets/image-20221128173038275.png)

## Multi-Class  

> ![image-20221128173211019](P_Homework4_叶璨铭.assets/image-20221128173211019.png)

## Convex Hull

> ![image-20221128173313881](P_Homework4_叶璨铭.assets/image-20221128173313881.png)

