from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras.models import Model

import numpy as np

#def create_encoding(model, layerid):
#    outputs = [layer.output for layer in model.layers]
#    intermediate_layer_model = Model(inputs=model.input, outputs=outputs[layerid])
#    return intermediate_layer_model

(x_train, y_train), (x_test,y_test) = mnist.load_data()

# Preprocessing
x_train = x_train / 255.0
x_test = x_test / 255.0
#x_train=np.expand_dims(x_train,axis=3)
#x_test=np.expand_dims(x_test,axis=3)
x_train_reshaped = np.reshape(x_train, (60000, 28*28))
x_test_reshaped = np.reshape(x_test, (10000, 28*28))

y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test,10)

#model.add(Dense(500, input_shape=(784,), activation='sigmoid'))


#model.add(LSTM(100, input_shape=(28, 28)))
#model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
#model.add(Conv2D(filters=4, kernel_size=(3,3), padding='same', activation='relu'))
#model.add(Flatten())
#model.add(Dense(100, activation='sigmoid'))
#model.add(Dense(250, activation='sigmoid'))
#model.add(Dense(500, activation='sigmoid'))
#model.add(Dense(10, activation='sigmoid'))

model = Sequential()
model.add(Dense(500, activation='sigmoid',input_shape=(784,)))
model.add(Dense(2000, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam')

save_model_name = os.path.join('./models/model_classifier.h5')
saveBest = ModelCheckpoint(save_model_name, monitor='val_loss', save_best_only=True)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=2, mode='auto')

model.fit(x_train_reshaped, y_train_categorical, batch_size=1,epochs=50,verbose=1,validation_split=0.05, callbacks=[saveBest,earlyStopping])

