
from keras.models import *
from keras.layers import *

def create_model(sample_size):
    model = Sequential()
    
    model.add(Conv1D(32, 3, activation='relu', input_shape=(sample_size, 56)))
    model.add(MaxPooling1D())

    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D())

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D())

    model.add(LSTM(128))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    
    return model
