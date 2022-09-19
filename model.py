import torch
from keras.layers import Activation, Dropout, BatchNormalization
from keras.layers import LSTM, ConvLSTM1D, ConvLSTM2D, Dense, Conv2D
from keras.models import Sequential
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(2)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        pred = self.softmax(pred)
        pred = pred[:, -1, :]  # (5, 1)
        return pred


def lstm(L, n_features):
    model = Sequential()

    model.add(LSTM(input_shape=(L, n_features), units=128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(units=128, return_sequences=False))
    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def convlstm(L, n_features):
    model = Sequential()

    model.add(ConvLSTM1D(input_shape=(1, L, n_features), filters=128, kernel_size=3, return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM1D(filters=128, kernel_size=3, return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM1D(filters=1, kernel_size=L-4, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def conv(L, n_features):
    model = Sequential()

    model.add(Conv2D(input_shape=(L, n_features, 1), filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(L, n_features)))

    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
