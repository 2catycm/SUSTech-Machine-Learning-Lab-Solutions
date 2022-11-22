$$
\begin{align}
J     =
      \left(\begin{array}{cc}
      \frac{\partial \bf{y} }{\partial x_{1} } &
      ... &
      \frac{\partial \bf{y} }{\partial x_{n} }
      \end{array}\right)
     =
     \left(\begin{array}{ccc}
      \frac{\partial y_{1} }{\partial x_{1} } & \cdots & \frac{\partial y_{1} }{\partial x_{n} }\\
      \vdots & \ddots & \vdots\\
      \frac{\partial y_{m} }{\partial x_{1} } & \cdots & \frac{\partial y_{m} }{\partial x_{n} }
      \end{array}\right)
\end{align} 
$$


Generally speaking, ``torch.autograd`` is an engine for computing
vector-Jacobian product. That is, given any vector $\vec{v}$, compute the product
$J^{T}\cdot \vec{v}$

If $\vec{v}$ happens to be the gradient of a scalar function $l=g\left(\vec{y}\right)$:

$$
\begin{align}\vec{v}
   =
   \left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}\end{align}
$$

then by the chain rule, the vector-Jacobian product would be the
gradient of $l$ with respect to $\vec{x}$:

$$
\begin{align}J^{T}\cdot \vec{v}=\left(\begin{array}{ccc}
      \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{1}}\\
      \vdots & \ddots & \vdots\\
      \frac{\partial y_{1}}{\partial x_{n}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
      \end{array}\right)\left(\begin{array}{c}
      \frac{\partial l}{\partial y_{1}}\\
      \vdots\\
      \frac{\partial l}{\partial y_{m}}
      \end{array}\right)=\left(\begin{array}{c}
      \frac{\partial l}{\partial x_{1}}\\
      \vdots\\
      \frac{\partial l}{\partial x_{n}}
      \end{array}\right)\end{align}
$$
This characteristic of vector-Jacobian product is what we use in the above example;
``external_grad`` represents $\vec{v}$.


