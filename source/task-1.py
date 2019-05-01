# datsets
import sklearn.datasets as datasets
import pandas as pd
# for the NN
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
# for changing the optimizer
from keras.optimizers import adam
# for the tensorboard graph
from keras.callbacks import TensorBoard
# for the random predictions at the end
import random

# load data
data = datasets.load_boston();
X = pd.DataFrame(data.data, columns=data.feature_names)

# train test split
X_train = X.values[:400, :]
Y_train = data.target[:400]

X_test = X.values[401:, :]
Y_test = data.target[401:]

# custom adam optimizer and other meta parameters
adam = adam(lr=0.002)
epochnum = 150
batchsize = 20
activationfn = 'tanh'

noLoad = input("load model? y/n ").lower() == 'n'

# try to load network
try:
    if noLoad == True:
        raise
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    model.compile(loss='mean_squared_error', optimizer=adam)
    print("loaded")

except:
    # create network
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation=activationfn))
    model.add(Dense(10, activation=activationfn))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=adam)

    tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # fit on training data
    model.fit(X_train, Y_train, epochs=epochnum, batch_size=batchsize, verbose=2, callbacks=[tbCallBack])

    # save model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

# evaluate on test data
ev = model.evaluate(X_test, Y_test, verbose=2)
print("Testing results: {0} loss on test set".format(ev))

# show random predictions (just to show how the network learned)
print("Sample#: Predicted|Actual")

r1 = random.randint(0, 505)
test = X.values[r1, :]
pred1 = round(model.predict(test.reshape(1, 13)).item(), 2)
act1 = data.target[r1]
print("{0}:\t{1}|{2}".format(r1, pred1, act1))

r2 = random.randint(0, 505)
test = X.values[r2, :]
pred2 = round(model.predict(test.reshape(1, 13)).item(), 2)
act2 = data.target[r2]
print("{0}:\t{1}|{2}".format(r2, pred2, act2))

r3 = random.randint(0, 505)
test = X.values[r3, :]
pred3 = round(model.predict(test.reshape(1, 13)).item(), 2)
act3 = data.target[r3]
print("{0}:\t{1}|{2}".format(r3, pred3, act3))

# pause for terminal
input("press enter to continue...")