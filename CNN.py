from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, GaussianNoise,\
    SpatialDropout2D, Activation
from keras.regularizers import l1


class CNN:
    def __init__(self, image_size):
        self.image_size = image_size
        self.model = 0

    def build(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=self.image_size, padding='same'))
        self.model.add(GaussianNoise(0.1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(SpatialDropout2D(0.3))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(GaussianNoise(0.1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(SpatialDropout2D(0.2))

        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(SpatialDropout2D(0.2))

        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(1, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', activity_regularizer=l1(0.001)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
