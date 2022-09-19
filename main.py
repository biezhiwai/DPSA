import numpy as np
import pandas as pd
import tqdm
from torch.optim.lr_scheduler import StepLR

from dataset import load_data, generate_dataset
import gc
from keras import backend
from model import lstm, conv, convlstm, LSTM
from KTFCM import generate_points, ktfcm
import torch


if __name__ == '__main__':
    data = pd.read_csv("data/深证A指_processed.csv", index_col=0)  # 读取经过预处理的数据集
    Ntr = 3333  # 训练集样本数
    Ntu = 40  # 调优集样本数
    Nte = 1  # 测试集样本数
    c = 5  # 聚类中心个数
    m = 2.0  # 模糊化系数
    sigma = 40  # 高斯核函数中，控制径向伸长的宽度参数
    delta = 0.7  # 权重衰减系数
    SL = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # 序列长度L的所有取值
    # 输入特征，有股票涨跌幅（放在最前面，用于聚类）、technical indicators、other indicators
    features = ['mov_percent', 'EMA-5', 'ROC-5', 'RSI']
    batch_size = 512
    epochs = 100
    n_windows = 30  # 要预测的天数

    predict_accuracy = 0.0  # 最终预测精度

    X = [load_data(data, L, Ntr + Ntu + Nte + n_windows - 1, features) for L in SL]  # 装载数据

    for i_window in range(n_windows):  # 遍历滑动窗口
        backend.clear_session()
        model = []  # 候选预测器集合
        model_accuracy = [0 for _ in range(len(SL))]  # 每个候选预测器的评估精度
        tests = []  # 不同L的测试集

        for i in range(len(SL)):
            samples = X[i]
            # 生成并划分数据集
            x_train, x_tune, x_test, y_train, y_tune, y_test = generate_dataset(samples, i_window, Ntr, Ntu, Nte)

            # 训练参数为Li的模型
            model.append(lstm(SL[i], len(features)))
            model[i].fit(
                x_train,
                y_train,
                batch_size=512,
                epochs=100,
                validation_split=0.05)

            # 构造聚类集
            x_cluster = np.row_stack((x_tune, x_test))
            # 生成样本点并聚类
            points = generate_points(x_cluster, c)
            ktfcm(points, c, m, delta)

            # 根据目标样本所属类别，从调优集中构造评估集
            x_evaluate = np.array([x_tune[i] for i in range(Ntu) if points[i].group == points[-1].group])
            y_evaluate = np.array([y_tune[i] for i in range(Ntu) if points[i].group == points[-1].group])
            # 计算参数为Li的模型的评估精度
            model_accuracy[i] = model[i].evaluate(x_evaluate, y_evaluate)[1]
            tests.append([x_test, y_test])
            print("*************模型的评估精度：")
            print(model_accuracy[i])

            del x_train, x_tune, x_test, y_train, y_tune, y_test, samples
            gc.collect()
            backend.clear_session()

        # 选出评估精度最高的模型，用它做最终预测，并更新预测精度
        i_max = np.argmax(model_accuracy)
        predict_accuracy += model[i_max].evaluate(tests[i_max][0], tests[i_max][1])[1]
        result = pd.DataFrame([predict_accuracy / (i_window + 1)])
        result.to_csv('result.csv')

    # 计算最终预测精度
    predict_accuracy /= n_windows
    print("##################################################     最终预测精度：", predict_accuracy)

    del X
    gc.collect()
    backend.clear_session()
