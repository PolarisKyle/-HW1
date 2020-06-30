import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 指定相对路径
path = "./Data/"

# 利用 pands 进行读取文件操作
train = pd.read_csv(path + 'train.csv', engine='python', encoding='big5')
test = pd.read_csv(path + 'test.csv', engine='python', encoding='gbk')
train = train[train['測項'] == 'PM2.5']
# print(train)
test = test[test['AMB_TEMP'] == 'PM2.5']
# 删除无关特征
train = train.drop(['日期', '測站', '測項'], axis=1)
# 读取test中第2列后的所有内容
test_x = test.iloc[:, 2:]

train_x = []
train_y = []

for i in range(15):
# x读取train中，第i列到i+8列的内容
    x = train.iloc[:, i:i + 9]
    # notice if we don't set columns name, it will have different columns name in each iteration
    x.columns = np.array(range(9))
# y读取train中，第i+9列内容
    y = train.iloc[:, i + 9]
    y.columns = np.array(range(1))
# 将x，y的内容添加到train_x和train_y中
    train_x.append(x)
    train_y.append(y)

# review "Python for Data Analysis" concat操作
# train_x and train_y are the type of Dataframe
# 取出 PM2.5 的数据，训练集中一共有 240 天，每天取出 15 组 含有 9 个特征 和 1 个标签的数据，共有 240*15*9个数据
# 相同字段的表首尾相接
train_x = pd.concat(train_x) # (3600, 9) Dataframe类型
train_y = pd.concat(train_y)

# 将str数据类型转换为 numpy的 ndarray 类型
train_y = np.array(train_y, float)
test_x = np.array(test_x, float)
# print(train_x.shape, train_y.shape)

# 进行标准缩放，即数据归一化
ss = StandardScaler()

# 进行数据拟合
ss.fit(train_x)
# 进行数据转换
train_x = ss.transform(train_x)

ss.fit(test_x)
test_x = ss.transform(test_x)

# 定义评估函数
# 计算均方误差（Mean Squared Error，MSE）
# r^2 用于度量因变量的变异中 可以由自变量解释部分所占的比例 取值一般为 0~1
def r2_score(y_true, y_predict):
    # 计算y_true和y_predict之间的MSE
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    # 计算y_true和y_predict之间的R Square即确定系数
    return 1 - MSE / np.var(y_true)

# 线性回归
class LinearRegression:

    def __init__(self):
        # 初始化 Linear Regression 模型
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        # 根据训练数据集X_train, y_train训练Linear Regression模型
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 对训练数据集添加 bias
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        '''
        :param X_train: 训练集
        :param y_train: label
        :param eta: 学习率
        :param n_iters: 迭代次数
        :return: theta 模型参数
        '''
        # 根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        # 定义损失函数
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')
        # 对损失函数求导
        def dJ(theta, X_b, y):
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(y)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            '''
            :param X_b: 输入特征向量
            :param y: lebel
            :param initial_theta: 初始参数
            :param eta: 步长
            :param n_iters: 迭代次数
            :param epsilon: 容忍度
            :return:theta：模型参数
            '''
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1]) # 初始化theta
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        # 给定待预测数据集X_predict，返回表示X_predict的结果向量
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        # 根据测试数据集 X_test 和 y_test 确定当前模型的准确度
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return

# 模型训练
LR = LinearRegression().fit_gd(train_x, train_y)
# 评分
LR.score(train_x, train_y)
# 预测
result = LR.predict(test_x)

# 结果保存
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', engine='python', encoding='big5')
sampleSubmission['value'] = result
sampleSubmission.to_csv(path + 'result.csv')
