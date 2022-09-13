import numpy as np
import random
import math
import copy
import pylab

FLOAT_MAX = 1e100


# 样本点
class Point:
    __slots__ = ["x", "group", "membership"]

    def __init__(self, x, c, group=0):
        self.x = x  # 该点的特征向量
        self.group = group  # 该点所属的组别
        self.membership = [0.0 for _ in range(c)]  # 该点关于各中心点的隶属度

    def deepcopy(self):  # 返回该对象的一个深拷贝
        return copy.deepcopy(self)


# 高斯核函数
def kernel_gaussian(point_a, point_b, sigma=40):
    similarity = np.exp(-np.linalg.norm(point_a.x - point_b.x) ** 2 / (sigma ** 2))
    return similarity


# 生成聚类点集
def generate_points(x_cluster, c):
    x_cluster = x_cluster.reshape((x_cluster.shape[0], -1))
    points = [Point(x, c) for x in x_cluster]  # 样本点集合
    return points


# 求距离某点最近的中心点
def get_nearest_center(point, center_points):
    i_min = point.group
    min_dis = FLOAT_MAX

    for i, center_point in enumerate(center_points):
        dis = kernel_gaussian(point, center_point)
        if dis < min_dis:
            min_dis = dis
            i_min = i
    return min_dis


# 初始化中心点集合
def initialize_center_points(points, c):  # k-means++的初始化方法
    n_points = len(points)
    center_points = [random.choice(points).deepcopy()]  # 先随机选择一个点
    distances = [0.0 for i in range(n_points)]
    sum_distances = 0.0

    for i in range(1, c):  # 遍历中心点
        for j in range(n_points):  # 遍历每个点
            distances[j] = get_nearest_center(points[j], center_points[:i])
            sum_distances += distances[j]

        sum_distances *= random.random()

        for j, distance in enumerate(distances):  # 轮盘法，距离越远的点被选中的概率越大
            sum_distances -= distance
            if sum_distances < 0:
                center_points.append(points[j].deepcopy())
                break
    return center_points


# 更新某点关于所有中心点的隶属度
def update_membership(point, center_points, m):
    c = len(center_points)
    distances = [kernel_gaussian(point, center_points[i]) for i in range(c)]

    try:  # 检查该点是否与某中心点重合
        i_coincide = distances.index(1)
    except ValueError:  # 该点未与任何中心点重合，按公式计算该点对于所有中心点的隶属度
        distances = [pow(1 - dis, -1 / (m - 1)) for dis in distances]
        sum_distances = sum(distances)
        point.membership = [dis / sum_distances for dis in distances]
    else:  # 重合
        point.membership = [0.0 for _ in range(c)]
        point.membership[i_coincide] = 1.0


# 目标函数，KTFCM算法就是最小化目标函数
def value_function(points, center_points, m):
    c = len(center_points)
    ans = 0.0
    for i in range(c):
        for point in points:
            distance = kernel_gaussian(point, center_points[i])
            ans += pow(point.membership[i], m) * (2 - 2 * distance)
    return ans


# KTFCM聚类算法
def ktfcm(points, c, m, delta):
    n_points = len(points)
    center_points = initialize_center_points(points, c)
    center_points_traces = [[center_point.x] for center_point in center_points]

    threshold = 0.0005
    change = FLOAT_MAX
    loss = FLOAT_MAX

    while change >= threshold:
        # 更新模糊划分矩阵
        for point in points:
            update_membership(point, center_points, m)

        # 更新聚类中心向量
        for i in range(c):
            time_weights = [pow(delta, np.log2(n_points - k)) for k in range(n_points)]
            distances = [kernel_gaussian(points[k], center_points[i]) for k in range(n_points)]
            numerator = [0.0 for _ in range(len(points[0].x))]
            denominator = 0.0
            for k in range(n_points):
                tmp = time_weights[k] * pow(points[k].membership[i], m) * distances[k]
                numerator += tmp * points[k].x
                denominator += tmp
            # 更新第i个聚类中心
            center_points[i].x = numerator / denominator
            center_points_traces[i].append(center_points[i].x)

        # 计算目标函数
        new_loss = value_function(points, center_points, m)
        change = loss - new_loss
        loss = new_loss

    for point in points:
        point.group = np.argmax(point.membership)
    return center_points_traces


# 测试聚类性能
if __name__ == '__main__':
    n_points = 2000
    c = 5
    m = 2
    radius = 10
    # 生成随机二维点
    points = [Point(np.zeros(2), 5) for _ in range(n_points)]
    for point in points[:int(n_points / 2)]:
        r = random.random() * radius
        angle = random.random() * 2 * math.pi
        point.x[0] = r * math.cos(angle)
        point.x[1] = r * math.sin(angle)
    for point in points[int(n_points / 2):]:
        point.x[0] = 2 * radius * random.random() - radius
        point.x[1] = 2 * radius * random.random() - radius
    center_points_traces = ktfcm(points, c, m, 1)
    # 可视化
    colors = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
    pylab.figure(figsize=(9, 9), dpi=80)
    for point in points:
        color = ''
        if point.group >= len(colors):
            color = colors[-1]
        else:
            color = colors[point.group]
        pylab.plot(point.x[0], point.x[1], color)
    for trace in center_points_traces:
        pylab.plot([x[0] for x in trace[:-1]], [x[1] for x in trace[:-1]], 'k')
    for trace in center_points_traces:
        pylab.plot(trace[-1][0], trace[-1][1], 'oy')
    pylab.show()
