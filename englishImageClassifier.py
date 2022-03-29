print('Setting Up...')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from tensorflow import keras
from keras import layers
from keras.optimizers import adam_v2
import pickle
import joblib
##########################################
path = 'myData'
# pathLabels = 'labels.csv'

testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)
batchSizeVal = 50
epochsVal = 8
# stepsPerEpochVal = 2000
##########################################
images = []
classNo = []
myList = os.listdir(path)
print("Total No of classes detected", len(myList))
noOfClasses = len(myList)
print("importing classes ... ")

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + '/' + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for i in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == i)[0]))
print(numOfSamples)

#### Visualization
# plt.figure(figsize=(10, 5))
# plt.bar(range(0, noOfClasses), numOfSamples)
# plt.title("No of images for each class")
# plt.xlabel("Class ID")
# plt.ylabel("Number of Images")
# plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
#
# img = preProcessing(X_train[100])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("Our preprocessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
#


dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes= 500

    model = keras.Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
#
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    opt = adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.compile(adam_v2(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#

print("X_train length: ", len(X_train))
print("batchsizeval: ", batchSizeVal)
# time.sleep(10000)



model = myModel()
print(model.summary())
#
#
stepsPerEpochVal = len(X_train)//batchSizeVal


history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                              steps_per_epoch=stepsPerEpochVal,
                              epochs=epochsVal,
                              validation_data=(X_validation, y_validation),
                              shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
#
# #### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])
#
# ### SAVE THE TRAINED MODEL
# pickle_out = open("model_trained.p", "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

#### SAVE using keras
model_json = model.to_json()
with open("englishModel.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("englishModel.h5")
print("Saved model to disk")

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)


# img = cv2.imread("9.jpg")
# # cv2.imshow("hi", img)
# img = np.asarray(img)
# img = cv2.resize(img, (32, 32))
# img = preProcessing(img)
# img = img.reshape(1, 32, 32, 1)
# pred = model.predict(img)
# classIndex = np.argmax(pred, axis=1)
# print(pred)
# print(classIndex)


