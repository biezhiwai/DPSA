import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

from dataset import load_data, generate_dataset
from model import LSTM

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    data = pd.read_csv("data/深证A指_processed.csv", index_col=0)  # 读取经过预处理的数据集
    Ntr = 3333  # 训练集样本数
    Ntu = 40  # 调优集样本数
    Nte = 1  # 测试集样本数
    # 输入特征，有股票涨跌幅（放在最前面，用于聚类）、technical indicators、other indicators
    features = ['mov_percent', 'EMA-5', 'ROC-5', 'RSI']
    batch_size = 512
    epochs = 100
    n_windows = 30  # 要预测的天数
    predict_accuracy = 0.0  # 最终预测精度

    samples = load_data(data, seq_len=10, n_samples=Ntr + Ntu + Nte + n_windows - 1, features=features)  # 装载数据

    for i_window in range(n_windows):
        # 生成并划分数据集
        x_train, x_tune, x_test, y_train, y_tune, y_test = generate_dataset(samples, i_window, Ntr, Ntu, Nte)

        # 训练模型
        train = torch.utils.data.TensorDataset(x_train, y_train)
        test = torch.utils.data.TensorDataset(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        model = LSTM(input_size=len(features),
                     hidden_size=128,
                     output_size=2,
                     num_layers=2,
                     batch_size=batch_size).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        scheduler = StepLR(optimizer)

        for epoch in tqdm(range(epochs)):
            train_loss = []
            for (seq, label) in train_loader:
                seq = seq.to(device)
                label = label.to(device)
                y_pred = model(seq)
                loss = loss_fn(y_pred, label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backkward()
                optimizer.step()
            scheduler.step()
            print('epoch {:03d} train_loss {:.8f}'.format(epoch, np.mean(train_loss)))
            model.train()
