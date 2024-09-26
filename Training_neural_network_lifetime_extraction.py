# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:56:35 2024

@author: lopez
"""

# We import the necessary libraries.
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf

# We load the training data.
df = pd.read_excel('D:/Microscope users/Sergio Lopez/Python/AI Capstone/Data constant IRF/training_constant_irf.xlsx', index_col=None)

# We transform the training data into a Numpy array.
df_array = df.values

# We assign the features and the outputs.
X = df_array[:,1:-2]
y = df_array[:,-2:]

# We define the keras model.
model = Sequential()
model.add(Input(shape=(512,))) # Input tensor
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='linear'))

# We compile the keras model
model.compile(loss='mean_squared_error', optimizer= Adam(learning_rate=0.0005), metrics=['mean_squared_error', 'mean_absolute_error'])

# We define the callbacks.
early_stop = EarlyStopping(monitor='val_loss', patience = 25, verbose=2)
log_csv = CSVLogger('Model_3_fit_lr00005.csv', separator= ',', append=False)
callbacks_list = [early_stop, log_csv]

# We fit the keras model on the dataset.
history = model.fit(X, y, validation_split=0.1, epochs=1000, batch_size=10, verbose=2, callbacks=callbacks_list)

# We plot the validation over time.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.semilogy()
plt.tight_layout()
plt.savefig('Model_loss.png',dpi=300)
plt.show()

# We plot the mean squared error over time. 
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Mean squared error')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.semilogy()
plt.tight_layout()
plt.savefig('Model_mse.png',dpi=300)
plt.show()

# We plot the mean absolute error over time. 
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean absolute error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.semilogy()
plt.tight_layout()
plt.savefig('Model_mae.png',dpi=300)
plt.show()

# We save the model.
model.save('Model_3_fit_lr00005.keras')