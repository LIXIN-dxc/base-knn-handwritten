import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

plt.rcParams['font.family'] = ['SimHei']  # 使用黑体

def img2vector(filename):
    """将32x32图像转换为1x1024向量"""
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline().strip()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def classify0(inX, dataSet, labels, k):
    """手动KNN分类器"""
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    distances = np.sqrt((diffMat**2).sum(axis=1))
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #返回计数最多对应的标签
    return max(classCount.items(), key=lambda x: x[1])[0]

def load_dataset(data_path):
    """加载数据集"""
    file_list = listdir(data_path)
    data_matrix = np.zeros((len(file_list), 1024))
    labels = []
    for i, filename in enumerate(file_list):
        labels.append(int(filename.split('_')[0]))
        #构建训练矩阵
        data_matrix[i] = img2vector(f'{data_path}/{filename}')
    return data_matrix, labels

# 主程序
train_data, train_labels = load_dataset('trainingDigits')
test_data, test_labels = load_dataset('testDigits')

# 手动KNN测试
manual_predictions = [classify0(test_data[i], train_data, train_labels, 3)
                     for i in range(len(test_data))]
manual_accuracy = accuracy_score(test_labels, manual_predictions)

# Sklearn KNN测试
sklearn_knn = KNN(n_neighbors=3)
sklearn_knn.fit(train_data, train_labels)
sklearn_accuracy = sklearn_knn.score(test_data, test_labels)

# 计算错误率
manual_error = 1 - manual_accuracy
sklearn_error = 1 - sklearn_accuracy

# 准备数据
methods = ['手动KNN', 'Sklearn KNN']
accuracies = [manual_accuracy * 100, sklearn_accuracy * 100]
error_rates = [manual_error * 100, sklearn_error * 100]

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 第一个子图：准确率对比
x = np.arange(len(methods))
width = 0.6  # 柱状图宽度

# 绘制准确率柱状图
bars1 = ax1.bar(x, accuracies, width, color=['lightblue', 'lightgreen'], alpha=0.8)
ax1.set_title('准确率对比')
ax1.set_ylabel('准确率(%)')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(0, max(accuracies) + 10)  # 为标注留出空间

# 在准确率柱状图上添加标注
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

# 第二个子图：错误率对比
bars2 = ax2.bar(x, error_rates, width, color=['lightcoral', 'lightsalmon'], alpha=0.8)
ax2.set_title('错误率对比')
ax2.set_ylabel('错误率(%)')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylim(0, max(error_rates) + 10)

# 在错误率柱状图上添加标注
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 2,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=10)

# 添加整体标题
fig.suptitle('手动KNN vs Sklearn KNN 性能对比', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"手动KNN - 正确率: {accuracies[0]:.2f}%, 错误率: {error_rates[0]:.2f}%")
print(f"Sklearn KNN - 正确率: {accuracies[1]:.2f}%, 错误率: {error_rates[1]:.2f}%")
