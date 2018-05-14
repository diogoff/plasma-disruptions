
from keras.models import *
from keras.layers import *

def create_model(sample_size):

    model = Sequential()
    
    model.add(Conv1D(32, 5, input_shape=(sample_size, 56)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())

    model.add(Conv1D(64, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    
    model.add(LSTM(200, return_sequences=True))
    model.add(LSTM(200))
    
    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(1))
    
    model.summary()
    
    return model
