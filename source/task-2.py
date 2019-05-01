#for datasets
import pandas as pd
import numpy as np
#for dataset normalization and train/test splitt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
#for NN
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import rmsprop
#for the tensorboard graph
from keras.callbacks import TensorBoard

#load dataset
dataset = pd.read_csv("heart.csv",header=0).values

#train/test split
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:12], dataset[:,13],
                                                    test_size=0.25, random_state=87)

#normalize data
X_train = normalize(X_train, norm='max',axis=0)
X_test = normalize(X_test, norm='max',axis=0)


#metaparameters (do the commented ones for the first run)
rms = rmsprop(lr=.002) #.001
numEpoch = 10 #100
activationFN = 'tanh' #relu

#simple logistic regression model
tbCallBack=TensorBoard(log_dir='./model1',histogram_freq=0,write_graph=True, write_images=True)

model = Sequential()
model.add(Dense(13, input_dim=X_train.shape[1], activation=activationFN))
model.add(Dense(5, activation=activationFN))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'] )
model.fit(X_train, Y_train, epochs=numEpoch, verbose = 2, callbacks = [tbCallBack])

#evaluate model
print(model.evaluate(X_test, Y_test, verbose=2))
