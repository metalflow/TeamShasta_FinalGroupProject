#imports
import time
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
#from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
#from sklearn import linear_model
np.random.seed(123)  # for reproducibility


#constants

#variables

#classes

#main program
outputFile = open("finalreportdata.csv","w")
outputFile.write("numEpochs,CNN_loss,CNN_acc,CNN_time,SVM_loss,SVM_acc,SVM_time\n")
for numEpochs in range(0,100,5):
    outputFile.write(str(numEpochs)+",")
    #load dataset
    (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    #print("x_train")
    #print(x_train.shape)
    #plt.imshow(x_train[0])
    #reshape the tensor to specify a single color channel (grayscale)
    print("x_train reshape")
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    #print(x_train.shape)
    #normalize values to floats between 0 and 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print("y_train")
    print(y_train.shape)
    print("first 10 items in y_train")
    print( y_train[:10] )
    print("y_train reshape")
    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print(y_train.shape)

    #begin creating model
    model = Sequential()
    model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    print( model.output_shape )
    model.add(Convolution2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #begin training
    tstart = time.time()
    model.fit(x_train, y_train,batch_size=32, epochs=numEpochs, verbose=0)
    tend = time.time()

    CNN_score = model.evaluate(x_test, y_test, verbose=0)
    #outputFile.write(str(CNN_score.get("loss"))+","+str(CNN_score.get("acc"))+","+str(tend-tstart))
    outputFile.write(str(CNN_score))
    outputFile.write(str(tend-tstart))
    #print("test loss, test acc:",CNN_score)
    #print("total time:",(tend-tstart))


    #clean up

    #begin SVM approximation
    #create the model
    model = keras.Sequential(
        [
            keras.Input(shape=(784,)),
            RandomFourierFeatures(
                output_dim=4096, scale=10.0, kernel_initializer="gaussian"
            ),
            layers.Dense(units=10),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.hinge,
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    #prepare the data
    #load dataset
    (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
    # Preprocess the data by flattening & scaling it
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    # Categorical (one hot) encoding of the labels
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    tstart = time.time()
    model.fit(x_train, y_train, epochs=numEpochs, batch_size=128, validation_split=0.2,verbose=0)
    tend = time.time()
    #print("SVM_score")
    SVM_score = model.evaluate(x_test, y_test, verbose=0)
    #outputFile.write(str(CNN_score.get("loss"))+","+str(CNN_score.get("acc"))+","+str(tend-tstart))
    outputFile.write(str(SVM_score))
    outputFile.write(str(tend-tstart))
    #print("test loss, test acc:",SVM_score)
    #print("total time:",(tend-tstart))
    outputFile.write("\n")



##logistic regression (copied from Teo's repository)

#(Xtrain,ytrain),(Xtest,ytest) = fashion_mnist.load_data()

#ntrain = Xtrain.shape[0]
#ntest = Xtest.shape[0]
#nrow = Xtrain.shape[1]
#ncol = Xtrain.shape[2]


#npixels = nrow*ncol
#Xtrain = 2*(Xtrain/255 - 0.5)
#Xtrain = Xtrain.reshape((ntrain,npixels))

#Xtest = 2*(Xtest/255 - 0.5)
#Xtest = Xtest.reshape((ntest,npixels))


#model = linear_model.LogisticRegression(verbose=0, solver='sag', max_iter=10)
#tstart = time.time()
#model.fit(Xtrain,ytrain)
#tend = time.time()

#prediction = model.predict(Xtest)
#successrate = np.mean(prediction == ytest)
#print('Accuracy: {0:f}'.format(successrate))
#print("total time:",(tend-tstart))