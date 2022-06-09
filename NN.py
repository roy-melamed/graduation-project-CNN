from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


class NN:
    def __init__(self, data_size):
        self.data_size = data_size
        self.model = 0

    def build(self):
        self.model = Sequential()
        self.model.add(Dense(2048, kernel_initializer='random_normal', activation='relu', input_shape=self.data_size))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1024, kernel_initializer='random_normal', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(512, kernel_initializer='random_normal', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(512, kernel_initializer='random_normal', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256, kernel_initializer='random_normal', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, kernel_initializer='random_normal', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, kernel_initializer='random_normal', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(32, kernel_initializer='random_normal', activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

