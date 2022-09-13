import numpy as np
import pandas as pd
from keras.utils import np_utils


# 所有区间均为左闭右开
def load_data(data, L, n_samples, features):
    samples = []

    for i in range(len(data) - L - n_samples, len(data) - L):  # 枚举长度为L的所有序列
        sample = {'x': data.get(features).values[i: i + L],  # L天的特征向量组成一个样本(timesteps, n_features)
                  'y': data['movement'].iloc[i + L],  # 标签
                  'date': data['date'].iloc[i + L]}  # 该样本要预测涨跌的日期
        samples.append(sample)

    return samples


def generate_dataset(samples, i_window, Ntr, Ntu, Nte):
    # 数据集尺寸
    x_shape = (samples[0]['x'].shape[0], samples[0]['x'].shape[1])
    y_shape = (2,)
    # 用于划分数据集的下标位置
    train_begin = i_window * Nte
    tune_begin = train_begin + Ntr
    test_begin = tune_begin + Ntu
    test_end = test_begin + Nte
    # 打印要预测的股票走势的日期
    for sample in samples[test_begin:test_end]:
        print("--------------------     test day %d:" % i_window)
        print(sample['date'], "   sequence length:", len(sample['x']), "\n\n\n\n\n")

    # 生成数据集
    x = np.array([sample['x'] for sample in samples])
    x_train = np.reshape(x[train_begin:tune_begin], (Ntr,) + x_shape)
    x_tune = np.reshape(x[tune_begin:test_begin], (Ntu,) + x_shape)
    x_test = np.reshape(x[test_begin:test_end], (Nte,) + x_shape)

    y = np.array([sample['y'] for sample in samples])
    y = np_utils.to_categorical(y, num_classes=2)
    y_train = np.reshape(y[train_begin:tune_begin], (Ntr,) + y_shape)
    y_tune = np.reshape(y[tune_begin:test_begin], (Ntu,) + y_shape)
    y_test = np.reshape(y[test_begin:test_end], (Nte,) + y_shape)

    return x_train, x_tune, x_test, y_train, y_tune, y_test
