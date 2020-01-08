import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

def autoencoder_model(X):
    '''
    Autoencoder network model
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# Define arguments
parser = argparse.ArgumentParser(description='Time Series Anomaly Detection Argument Parser')
parser.add_argument('--data_dir', type=str, help='Directory where data is stored')
args = parser.parse_args()

# Get arguments from parser
data_dir = args.data_dir

# Load training data
train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'), index_col=0)
train_data.index = pd.to_datetime(train_data.index, format='%Y-%m-%d %H:%M:%S')

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data)
joblib.dump(scaler, './outputs/scaler_data')

# Reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# Create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')

# Fit the model to the data
nb_epochs = 100
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.05).history
model.save('./outputs/model.h5')

# Plot the loss distribution and save image to outputs
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train_data.columns)
X_pred.index = train_data.index

scored = pd.DataFrame(index=train_data.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([0.0,.2])
plt.savefig('./outputs/loss_distribution.png')
