"""
多项式回归是多元线性回归的特殊情况，它用多项式特征处理非线性数据。
"""
import numpy as np
import itertools
import functools


class PolynomialFeatures:
    """
    polynomial features
    transforms input array with polynomial features
    Example
    =======
    x =
    [[a, b],
    [c, d]]
    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """

    def __init__(self, degree=2):
        """
        construct polynomial features
        Parameters
        ----------
        degree : int
        degree of polynomial
        """
        self.degree = degree

    def fit_transform(self, X):
        """
        transforms input array with polynomial features
        Parameters
        ----------
        X : (sample_size, n) ndarray
        input array
        Returns
        -------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
        polynomial features
        """
        features = [np.ones(len(X))]
        X = X[np.newaxis, :]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(X, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()


class LinearRegression:
    """
    Linear regression model
    y = X @ w
    t ~ N(t|X @ w, var)
    """

    def __init__(self):
        self.w = None
        # self.var = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        perform least squares fitting
        Parameters
        ----------
        X : (N, D) np.ndarray
        training independent variable
        t : (N,) np.ndarray
        training dependent variable
        """
        self.w = np.linalg.inv(X.T@X)@X.T@t
        return self

    def predict(self, X: np.ndarray):
        """
        make prediction given input
        Parameters
        ----------
        X : (N, D) np.ndarray
        samples to predict their output
        Returns
        -------
        y : (N,) np.ndarray
            prediction of each sample
        """
        return X@self.w


def rmse(a, b):
    """root mean square error. 
    Args:
        a (vec<float>): 一列结果
        b (vec<float>): 另一列结果

    Returns:
        _type_: 两个结果的差的平方和的平均值的算术平方根，类似于标准差。
    """
    return np.sqrt(np.mean(np.square(a - b)))


def main():
    """从标准输入读入，计算后从标准输出输出结果。
    input:
        - N(int): 训练集样本量
        - M(int): 测试集样本量
        - x_train y_train (List<N, Tuple<float, float>>): 训练集
        - x_test y_test (List<M, Tuple<float, float>>): 测试集
    output:
        - K(int) in [0, 10] 多项式的阶数。选择使得测试集上标准差最小的K。
        - S(%.6f) 标准差 
    """
    n, m = input().split()
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for _ in range(int(n)):
        x, y = input().split()
        x_train.append(float(x))
        y_train.append(float(y))
    for _ in range(int(m)):
        x, y = input().split()
        x_test.append(float(x))
        y_test.append(float(y))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    reg = LinearRegression()
    test_errors = []
    train_errors = []
    for k in range(11):
        poly = PolynomialFeatures(k)
        reg.fit(poly.fit_transform(x_train), y_train)
        y_train_pred = reg.predict(poly.fit_transform(x_train))
        y_test_pred = reg.predict(poly.fit_transform(x_test))
        test_errors.append(rmse(y_test_pred, y_test))
        train_errors.append(rmse(y_train_pred, y_train))
    test_errors = np.array(test_errors)
    best_k = np.argmin(test_errors)
    print(f"{best_k}\n{train_errors[best_k]:6f}")
    # print(f'train_errors: {train_errors}')
    # print(f'test_errors: {test_errors}')
if __name__ == '__main__':
    main()
