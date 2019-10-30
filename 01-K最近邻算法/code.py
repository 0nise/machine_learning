from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

'''
K最近邻算法生成数据，并且通过matplotlib进行展示
'''
def knn_test_demo():
    data = make_blobs(n_samples=1000, centers=10, random_state=8)
    x, y = data
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.spring, edgecolors="k")
    plt.show()

#knn_test_demo()

'''
K最近邻算法，分类任务
'''
def knn_test_demo_():
    # 创建数据集
    data = make_blobs(n_samples=1000, centers=10, random_state=8)
    x, y = data
    # 创建KNN分类器
    clf = KNeighborsClassifier()
    clf.fit(x, y)
    # 绘制图
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, z)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.spring, edgecolors="k")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("KNN")
    #plt.scatter(6.95, 5.2, marker="*", c='red', s=200)
    plt.show()
    print('=' * 30)
    print("模型正确率：{:2f}".format(clf.score(x,y)))
   # print('新的数据点为：（6.95，5.2），分类结果为：', clf.predict([[6.95, 5.2]]))
    print('=' * 30)

'''
KNN 回归
'''
def knn_regressor_demo():
    from sklearn.datasets import make_regression
    # KNN 回归模型
    from sklearn.neighbors import KNeighborsRegressor
    x, y = make_regression(n_features=1, n_informative=1, noise=50, random_state=8)
    reg = KNeighborsRegressor(n_neighbors=2)
    reg.fit(x, y)
    z = np.linspace(-3, 3, 200).reshape(-1, 1)
    plt.scatter(x, y, c='orange', edgecolors="k")
    plt.plot(z, reg.predict(z), c='k', linewidth=3)
    plt.title('KNN Regressor')
    plt.show()
    print('=' * 30)
    print("模型正确率：{:2f}".format(reg.score(x, y)))
    # print('新的数据点为：（6.95，5.2），分类结果为：', clf.predict([[6.95, 5.2]]))
    print('=' * 30)

def read_wine_dataset():
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    win_dataset = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(win_dataset['data'], win_dataset['target'], random_state=0)
    print('X_train shape:{}'.format(X_train.shape))
    print('X_test shape:{}'.format(X_test.shape))
    print('y_train shape:{}'.format(y_train.shape))
    print('y_test shape:{}'.format(y_test.shape))

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def project():
    knn = KNeighborsClassifier(n_neighbors=1)
    win_dataset = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(win_dataset['data'], win_dataset['target'], random_state=0)
    knn.fit(X_train, y_train)
    print(knn)
    X_new = np.array([[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]])
    production = knn.predict(X_new)
    print('预测红酒的类别为', win_dataset['target_names'][production])
    # print('测试集样本预测', knn.score(X_test, y_test))