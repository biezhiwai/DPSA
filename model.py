from keras.layers import Activation, Dropout, BatchNormalization
from keras.layers import LSTM, ConvLSTM1D, ConvLSTM2D, Dense, Conv2D
from keras.models import Sequential


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
