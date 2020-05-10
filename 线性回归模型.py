import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *


# # 单变量（特征）线性模型
# class SingleVariableLinearRegression:
#     def __init__(self):
#         self.w = random.random()  # 初始化权重-实值
#         self.b = random.random()  # 初始化偏置-实值
#         self.cost = dict() # 记录代价变化
#
#     # 模型的训练
#     def train_model(self, x, y_true):
#         """
#         模型形式为：wx+b=y
#         :param x: 样本集--列向量-numpy数组，m*1,每行为一个样本
#         :param y_true: 预测值--实值 m*1
#         :return: 模型自身，参数已更新
#         """
#         self.w, self.b = self.optimization_method(x, y_true, self.w, self.b)
#         return self
#
#     # 前向计算
#     @staticmethod
#     def forward_calculation(x, w, b):
#         y = w*x+b
#         return y
#
#     # 目标函数/最小化损失函数
#     @staticmethod
#     def loss_function(y_true, y_pre):
#         return (np.square(y_true - y_pre)).mean(axis=0)  # 均方误差
#
#     # 优化方法
#     def optimization_method(self, x, y_true, w, b, alpha=0.01, max_iter=100):
#         # gradient_descent 梯度下降法：设置最大迭代次数和学习率
#         m = len(x)
#         for i in range(max_iter):
#             y_pre = self.forward_calculation(x, w, b)
#             single_cost = self.loss_function(y_true, y_pre)
#             self.cost[i] = single_cost
#             err = y_pre - y_true  # 误差矩阵
#             gradient = 1.0 / m * x.T.dot(err)  # 梯度矩阵
#             w -= alpha * gradient
#             b -= alpha * gradient
#         return w, b
#
#     # 预测
#     def predict(self, x):
#         return self.w * x + self.b


#################################################################
# 多变量（特征）线性模型
class MultiVariableLinearRegression:
    def __init__(self, features_num):
        self.w = np.random.randn(features_num, 1)  # 初始化权重--列向量矩阵，d*1,每一行为一个特征的系数
        self.b = np.random.random()  # 初始化偏置-实数
        self.cost = dict()  # 记录代价变化

    # 模型的训练
    def train_model(self, x, y_true):
        """
        模型形式为：wT*x+b=y
        :param x: 样本集--矩阵-d*m,每一列为一个样本，每一行代表一个特征
        :param y_true: 预测值--列向量--m*1,每一行表示一个样本
        :return: 模型自身，参数已更新
        """
        self.w, self.b = self.optimization_method(x, y_true, self.w, self.b)
        return self

    # 前向计算
    @staticmethod
    def forward_calculation(x, w, b):
        y = np.dot(w.T, x) + b
        return y

    # 目标函数/最小化损失函数
    @staticmethod
    def loss_function(y_true, y_pre):
        m = len(y_true)
        return (1.0 / (2 * m)) * np.square(y_true - y_pre).sum()

    # 优化方法
    def optimization_method(self, x, y_true, w, b, alpha=0.00002, max_iter=10000):
        # gradient_descent 梯度下降法：设置最大迭代次数和学习率
        d, m = x.shape
        for i in range(max_iter):
            y_pre = self.forward_calculation(x, w, b)
            single_cost = self.loss_function(y_true, y_pre)
            self.cost[i] = single_cost
            err = y_pre - y_true  # 误差矩阵
            gradient = (1.0 / m) * (np.sum(x.dot(err.T), ))  # 梯度矩阵

            w -= alpha * gradient
            b -= alpha * gradient
        return w, b

    # 预测
    def predict(self, x):
        return self.w * x + self.b


if __name__ == '__main__':
    train_datas = pd.read_csv(r'E:\python_work\machine_learning\datas\creditcard.csv')
    train_x = np.array(train_datas.iloc[1:100, 1:3].T)
    train_y = np.array(train_datas.iloc[1:100, -2]).T
    feature_num = train_x.shape[0]
    # A = MultiVariableLinearRegression(feature_num)
    # A.train_model(train_x, train_y)
    # # plt.scatter(train_x[0], train_x[1])
    # plt.plot(list(A.cost.keys()),list(A.cost.values()))
    # # x, y = np.meshgrid(train_x[0], train_x[1])
    # # print(A.w, A.b)
    # # def f(x, y):
    # #     z = A.w[0] * x + A.w[1] * y + A.b
    # #     return z
    # # plt.contour(x, y, f(x, y), 10)
    # plt.show()
    ################################
    from sklearn import linear_model  # 表示，可以调用sklearn中的linear_model模块进行线性回归。
    model = linear_model.LinearRegression()
    model.fit(train_x.T, train_y.T)
    y_p = model.predict(train_x.T)
    plt.plot(train_x.T, y_p)
    plt.show()