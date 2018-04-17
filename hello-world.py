from __future__ import print_function
import csv
from random import randint
import numpy as np
import math
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras import losses
import matplotlib.pyplot as plt

# features = input (vektor av inputs for hver dag)
# labels = output
# legg til index open/close
# legg til oljepris (?)

useVolume = False

dates = []

openPrice = []
high = []
low = []
closePrice = []
volume = []
# adjClose = []


lookbackDays = 10
trainingSplit = 0.2
volumeScalingFactor = 1 / 30000000
priceFactorDenom = 200
priceFactor = 1 / priceFactorDenom

x_test = []
x_train = []
y_test = []
y_train = []


def loadCsv(fileName):
    with open(fileName, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(row[0])
            openPrice.append(float(row[1]) * priceFactor)
            high.append(float(row[2]) * priceFactor)
            low.append(float(row[3]) * priceFactor)
            closePrice.append(float(row[4]) * priceFactor)
            # adjClose.append(float(row[5]) * priceFactor)
            volume.append(int(row[6]) * volumeScalingFactor)
    return


def isInTestSet(testFraction):
    testNumbers = testFraction * 100
    randomnumber = randint(1, 100)
    return testNumbers >= randomnumber



loadCsv("STL.OL.csv")
nrDays = len(openPrice)

for i in range(lookbackDays, nrDays):
    result = [low[i], closePrice[i]]
    inputs = [openPrice[i]]
    for j in range(1, lookbackDays+1):
        if useVolume:
            inputs.extend([openPrice[i - j], high[i - j], low[i - j], closePrice[i - j], volume[i - j]])
        else:
            inputs.extend([openPrice[i - j], high[i - j], low[i - j], closePrice[i - j]])
    if isInTestSet(trainingSplit):
        x_test.append(inputs)
        y_test.append(result)
    else:
        x_train.append(inputs)
        y_train.append(result)

# ML model
input_dim = 1 + lookbackDays * (4 + (1 if useVolume else 0))
num_output = 2
epochs = 25000
batch_size = 20

x_train_fit = np.array([np.array(xi) for xi in x_train])
x_test_fit = np.array([np.array(xi) for xi in x_test])
y_train_fit = np.array([np.array(xi) for xi in y_train])
y_test_fit = np.array([np.array(xi) for xi in y_test])

layerNodes = math.floor(input_dim * 0.7)

if layerNodes < 10:
    layerNodes = input_dim

#TODO sjekk om input/output dimensjoner er riktige

model = Sequential()
model.add(Dense(layerNodes, activation='relu', input_shape=(input_dim,)))
#model.add(Dropout(0.2))
model.add(Dense(layerNodes, activation='relu'))
model.add(Dense(layerNodes, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(num_output, activation='linear'))

model.summary()


model.compile(loss=losses.mean_squared_error,
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train_fit, y_train_fit,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test_fit, y_test_fit))
score = model.evaluate(x_test_fit, y_test_fit, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x_test_fit)
testSize = len(predictions)

predicted_lows = []
predicted_closes = []
test_results = []

for i in range(0, testSize-1):
    prediction = predictions[i]
    res = y_test_fit[i]

    low_predicted = prediction[0] * priceFactorDenom
    close_predicted = prediction[1] * priceFactorDenom
    low_actual = res[0] * priceFactorDenom
    close_actual = res[1] * priceFactorDenom

    if low_actual <= low_predicted:
        res = close_actual / low_predicted
        test_results.append(res)
    else:
        test_results.append(1)

    print('Low: ', low_predicted, '/', low_actual, ' - Close: ', close_predicted, '/', close_actual)

totalRes = 1
for i in range(0, testSize-1):
    totalRes = totalRes * test_results[i]
    print('%s Total res: %s' % (test_results[i], totalRes))

print('Total res:', totalRes)

#hvis test_result innerholder mange 1 så burde det kanskje velges cutoff lavere enn estimert