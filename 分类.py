import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from pylab import mpl
import random
import pandas as pd

# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
sns.set_theme()
plt.rcParams['font.family'] = 'SimHei'  # 替换为你选择的字体


def plot_feature_y(X, X_label, Y):
    """
    desc: 特征量与真实值的相关性
    """
    m, n = X.shape

    axs = []
    # 设置画布
    fig = plt.figure(figsize=(14, 14), dpi=100)
    plt.subplots_adjust(bottom=0, right=0.8, top=1, hspace=0.5)
    # 列
    coloum = 3
    for i in range(n):
        ax = fig.add_subplot(math.ceil(n / coloum), coloum, i + 1)
        if i == 0:
            ax.set_ylabel('真实值 y')
        ax.set_xlabel('x')
        ax.set_title(X_label[i])

        # 绘制散点图
        ax.scatter(X[:, i], Y)
        # 绘制箱型图
        np.random.seed(10)  # 设置种子
        D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))
        ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                   showmeans=False, showfliers=False,
                   medianprops={"color": "white", "linewidth": 0.5},
                   boxprops={"facecolor": "C0", "edgecolor": "white",
                             "linewidth": 0.5},
                   whiskerprops={"color": "C0", "linewidth": 1.5},
                   capprops={"color": "C0", "linewidth": 1.5})

        axs.append(ax)


def plot_cost(cost):
    """
    desc:绘制损失值图
    """
    fig, ax = plt.subplots()
    ax.set_title("代价变化图")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost")
    plt.plot(cost)


def plot_corr(data_set):
    """
    desc:绘制变量间的相关性关系
    """
    # 计算变量间的相关系数
    corr = data_set.corr()

    f, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("变量之间的相关性系数值")
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, cmap="YlGn", ax=ax)

def Z_score_normalization(X):
    """
    desc:Z-score归一化
        公式：x* = ( x − μ ) / σ
    paremeters:
        X   np.array (m,n) 原始数据
    return:
        X_nor np.array (m,n)  归一化后的
    """
    # 计算样本的特征的均值和标准差
    Mu =    np.mean(X, axis=0)
    Sigma = np.std(X,  axis=0)

    # print(f"Mu = {Mu}")
    X_nor = (X - Mu) / Sigma

    return X_nor, Mu, Sigma


# 正则化后的回归模型
def data_dispose(data_set):
    """
    desc:  数据处理  利用pandas库进行处理 返回numpy对象
    parameters:
        data_set  pandas类型  数据集
    return
        X_dispose np.array （m, n）  处理后的特征量
        Y_dispose np.array  (m,1)    处理后的真实值
        X_labels   list    (,n)      特征标签

    """
    data_set = pd.DataFrame(data_set)

    # 提取特征量、特征标签、真实值
    X_dispose = data_set.iloc[:, :-1]  # 特征量 ：除最后一列外都为特征
    Y_dispose = data_set.iloc[:, -1]  # 真实值：最后一列为真实值
    X_labels = data_set.columns  # 特征标签

    # print(f"Y_dispose={np.array(Y_dispose,ndmin=2).T}")

    return np.array(X_dispose), np.array(Y_dispose, ndmin=2).reshape(-1, 1, ), X_labels


def data_set_slice(data_set):
    """
    desc:数据集划分 8 2原则
    paremeters:
       data_set pandas 数据集
    return:
       data_training_set pandas 训练集
       data_test_set     pandas 测试集
    """
    # 总样本数
    m = data_set.shape[0]

    # 训练集
    data_training_set = data_set[: math.ceil(0.8 * m)]
    # 测试集
    data_test_set = data_set[math.floor(0.8 * m):]

    return data_training_set, data_test_set


def init_W_b(X):
    """
    desc: 初始化w和b模型参数值
    parameters:
        X  np.array     （m, n）  特征数据
    return:
        W  np.array     （n,1）    模型参数值
        b  float                  模型参数指
    """
    n = np.array(X).shape[1]

    W = np.zeros((n, 1))
    b = 0.
    print(f"初始化 b ={b}")
    return W, b


def sigmod(z):
    """
    desc:激活函数

    """
    f = 1 / (1 + np.exp(-z))

    return f


def Hypothesis_function(X, W, b):
    """
    desc: 假设函数
    parameters:
        X  np.array     （m, n）  特征数据
        W  np.array     （n,1）    模型参数值
        b  float                  模型参数指
    returns:
        f_wb  np.array  (m,1)    预测值
    """

    # 线性模型
    h_wb = X @ W + b

    # sigmod函数激活
    f_wb = sigmod(h_wb)

    # print(f"f_wb = {f_wb}")
    return f_wb


def regularize_lambda():
    """
    desc:给出正则系数
    lambda$大  ，则W_j小,惩罚大
    lambda$小  ，则W_j大
    return:
        _lambda  float  正则系数
    """
    _lambda = 0

    return _lambda


def cost_function(X, Y, W, b):
    """
    desc：代价函数
    parameters:
        X  np.array （m, n）    特征数据
        Y  np.array  (m,1)     真实值
        W  np.array  (n,1)     模型参数值
        b  float                模型参数值
    return:
        J_w_b  float            成本/代价
        Err    np.array  (m,1)  损失
    """
    m = np.array(X).shape[0]


    # 代价cost
    f_wb = Hypothesis_function(X, W, b)
    print(f"f_wb ={f_wb}")

    Err = f_wb - Y
    Loss = -Y * np.log(f_wb) - (1 - Y) * np.log(1 - f_wb)
    cost = (1 / m) * np.sum(Loss)

    # print(f"m= {m} f_wb={f_wb}")

    # 正则regularize
    _lambda = regularize_lambda()  # 正则系数
    regularize = (_lambda / (2 * m)) * np.sum(W ** 2)
    # 成本 = cost + regularize
    J_wb = cost + regularize

    return J_wb, Err


def compute_gradient_descent(X, Y, W, Err):
    """
    desc:计算正则化后的梯度(偏导)
    parameters:
        X  np.array （m, n）    特征数据
        Y  np.array  (m,1)      真实值
        W  np.array  (n,1)      模型参数值
        Err    np.array  (m,1)  损失
    return:
        dJ_dW np.array  (n,1)  J对w的偏导数
        dJ_db float            J对b的偏导数
    """
    m = np.array(X).shape[0]
    _lambda = regularize_lambda()

    # 计算偏导数
    tmp_dJ_dW = (1 / m) * np.dot(X.T, Err) + (_lambda / m) * W
    tmp_dJ_db = (1 / m) * np.sum(Err)

    # 同时更新
    dJ_dW = tmp_dJ_dW
    dJ_db = tmp_dJ_db

    return dJ_dW, dJ_db


def fit(X_train, Y_train, lr=0.01, iteration=10000):
    """
    desc:模型训练，模型拟合
    parameters:
        X_train  np.array （m, n）    训练集的特征数据
        Y_train  np.array  (m,1)      训练集的真实值
        lr  float  学习率 默认0.1
        iteration int 迭代次数 默认10000
    return:
        W_opt = W
        b_opt = b
    """
    # 数据处理
    X, Y = X_train, Y_train


    print(f"X={X.shape}  Y={Y.shape}")
    # 初始化模型参数
    W, b = init_W_b(X)

    # 损失
    Cost = []

    for index in range(iteration):
        # 1.计算cost，losss
        J_wb, Err = cost_function(X, Y, W, b)

        Cost.append(J_wb)
        ##############输出打印##############
        print(f"iteration {index+1}: cost = {J_wb}")


        # 2.计算梯度
        gradient_W, gradient_b = compute_gradient_descent(X, Y, W, Err)

        # 3.模型参数更新
        W -= lr * gradient_W
        b -= lr * gradient_b

    # 最优点
    W_opt = W
    b_opt = b

    # 绘制cost
    plot_cost(Cost)

    # 最小损失
    # 求列表最大值及索引
    min_value = min(Cost)  # 求列表最大值
    min_idx = Cost.index(min_value)  # 求最大值对应索引

    print(f"min cost: {min_value}  index={min_idx} iteration={iteration}")
    return W_opt, b_opt


def predict(X, W, b):
    """
    desc:模型预测
    parameters:
        X  np.array     （m, n）  特征数据
        W  np.array     （n,1）    模型参数值
        b  float                  模型参数指
    """
    predict_y = Hypothesis_function(X, W, b)

    return predict_y


def evaluate(data_test, W, b, mu=0, sigma=0):
    """
    desc:模型评价
    """
    # 数据处理
    X_test, Y_test, X_labels = data_dispose(data_test)

    # print(f"X_test={X_test} ")
    print("-----------test dataSet predict result---------")
    # 决策阈值
    threshold = 0.5

    # print(f"X_test={X_test}  {X_test.dtype}")
    # 预测
    predict = Hypothesis_function(X_test, W, b).reshape(-1).tolist()
    temp_pre = []
    # print(f"predict = {predict}")
    for i in range(len(predict)):
        if predict[i] > threshold:
            temp_pre.append(1)
        else:
            temp_pre.append(0)

    # 比较
    result = np.abs(np.array(temp_pre) - Y_test.T)

    # print(f"result={result}")
    # 统计正确个数
    err_count = np.sum(result)
    correct_count = len(predict) - err_count

    print(f"correct count: {correct_count}   error count: {err_count}")
    # 正确率
    correct_rate = correct_count / len(predict)

    return correct_rate, temp_pre, Y_test.T.tolist()



if __name__ == '__main__':
    # 加载数据集
    column_names = [
        'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
        'Uniformity of Cell Shape', 'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses', 'Class'
    ]
    data_set = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        names=column_names)


    # object转float
    data_set["Bare Nuclei"] = pd.to_numeric(data_set["Bare Nuclei"], errors='coerce')

    # 数据预处理:注意这里数据处理只能在上面一步把数据类型都转化为数字型的才能进行缺失值判断，因为判断函数仅认数字型
    # 1.缺失值处理：替换 or 删除
    data_set.fillna(0, inplace=True)
    # 丢弃带有缺失值的数据（只要有一个维度有缺失）删除带有nan的行
    data_set.dropna(axis=0, inplace=True)
    print(f"nan count ={data_set.isnull().sum()}")


    # 删掉编号列
    data_set.drop('Sample code number', axis=1, inplace=True)
    # 将标签替换成0 或 1
    min_value = data_set.iloc[:, -1].min()
    max_value = data_set.iloc[:, -1].max()
    data_set.iloc[:, -1].replace([min_value, max_value], [0, 1], inplace=True)






    # 数据集划分
    data_training_set, data_test_set = data_set_slice(data_set)

    # 训练集
    X, Y, X_label = data_dispose(data_training_set)

    # 绘制特征分量关于真实值的散点图
    plot_feature_y(X, X_label, Y)
    # 绘制变量相关性热力图
    plot_corr(data_set)
    # 规范化
    # X , mu , sigma =Z_score_normalization(X)

    # print(f"X={X} Y={Y}")
    # 数据训练拟合
    print(f"nan count ={len(X[np.isnan(X)])}")
    W_opt, b_opt = fit(X, Y, lr=0.01, iteration=10)
    print(f"最优参数：W={W_opt.reshape(-1)}   b={b_opt}")

    # mu = 0,
    # sigma = 0
    # # 模型评价：测试集
    # correct_rate, predict_y, y = evaluate(data_test_set, W_opt, b_opt, mu, sigma)
    # print("---------------------模型评价--------------------------")
    # print(f"correct rate: {correct_rate} ")
    # print(f"predict_y={predict_y}")
    # print(f"correct_y={y}")

import networkx as nx





class netGraph:
    """
    desc: 这是网络绘图类
    """
    def __init__(self, type=0):
        """
        desc:初始化函数
        """
        # 实例化一个NetworkX对象
        if type == 0:
            # 无向图
            self.G = nx.Graph()
        else:
            # 有向图
            self.G = nx.DiGraph()

        # 初始化节点数组:该类存储的节点，非networkX
        self.nodes_arr = []
    def netGraphToNetworkX(self):
        """
        desc: 将该类数据结构转化为NetWorkX中的对象
        """
        # 1.取nodes_arr数组中最后一个node进行转换
        last_node = self.nodes_arr[-1]  # 获取节点名称

        # 2.增加networkX中的node和edge
        self.G.add_node(last_node["name"])

        # 增加度
        for node in last_node["degree"]:
            if node["type"] == 0:
                # 入度
                self.G.add_edge(node["node"], node["name"], {'weight': node["weight"]})
            else:
                # 出度
                self.G.add_edge(node["name"], node["node"], {'weight': node["weight"]})
    def addNode(self, name="", desc="", pos=(1, 1), nexts=[{}], previous=[{}]):

        """
        desc:增加节点对象
        parmeters:
           name  str     节点名称  默认空白
           desc  str     节点描述  默认空白
           pos   tuple   节点在网络图中的位置 (层数n , 该层从上往下的序号j ) 1开始索引

           nexts     list  入度节点  默认空
           previous  list 入节点度  默认空白

           {
                "node": item["node"],#节点名称
                "label":  item["label"], #edge标签
                "weight":1 #权重
            }
        """

        # 度
        degrees = []
        for item in previous:
            degrees.append({
                "type": 0,  # 0 入度 1 为出度
                "node": item["node"],
                "label": item["label"],
                "weight": 1  # 权重
            })

        for item in nexts:
            degrees.append({
                "type": 1,  # 0 入度 1 为出度
                "node": item["node"],
                "label": item["label"],
                "weight": 1  # 权重
            })

        # 节点
        node = {
            "id": len(self.nodes_arr) + 1,  # 节点编号(自动编号)
            "name": name,  # 节点名称
            "desc": desc,  # 节点描述
            "position": self.setPostion(pos),  # 坐标位置(自动设置)
            "degree": degrees  # 度
        }
        # 添加进节点数组1中
        self.nodes_arr.append(node)

        # 调用节点转换
        self.netGraphToNetworkX()
    def setPostion(self, pos):
        """
        desc:节点坐标算法
        paremeters:
             pos   tuple   节点在网络图中的位置 (层数n , 该层从上往下的序号j ) 1开始索引
        """

        def line(pos):
            """
            desc: 直线性网络图坐标算法
            """
            # 初始化所在层的节点数
            n = 0

            # 遍历所在网络层的最大序号j
            for node in self.nodes_arr:
                # 相同层
                if node["position"][0] == pos[0]:
                    n += 1

            # 该节点的j
            j = n + 1

            # 间隔step
            step = 0.25

            # 坐标
            x = pos[0]
            y = (j - 1) * step + distance_before_layer(node["position"][0], step) / 2
            # 如果node_arr中没有该层节点        

            return (x, y)

        def distance_before_layer(l, step):
            """
            desc: 计算前一层l-1的节点总距离
            """
            # 计算l-1层的节点数
            n = 0
            for node in self.nodes_arr:
                # 相同层
                if node["position"][0] == (l - 1):
                    n += 1

            distance = n * step

            return distance

        return line(pos)

        def other():
            """
            desc:其他算法
            """
            pass

        # 节点位置:tuple  （层数， 序号）
        node_pos = pos

        # 节点坐标
        (x, y) = line(node_pos)

        return (x, y)
    def draw(self):
        """
        desc:绘图
        """
        self.G.draw()