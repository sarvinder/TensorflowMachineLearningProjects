
# coding: utf-8
"""This code is convolutional network for the image recognition on CFar10 data set
this is a 11 LAYERS NETWORK with 3 flatten layers including the final softmax layer
this only uses the batch normalization layer without any hyper parameter optimization

The optimized version will be out soon.
Training it on cpu will take around 3 hours  """


import keras

from keras.models import Sequential


from keras.layers import Dense,Conv2D,MaxPool2D,BatchNormalization,Dropout,Flatten

from keras.datasets import cifar10

##Know lets initialize the data and the the variables and hiperparameters
batch_size=32
num_classes=10
epochs=50

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

y_train= keras.utils.to_categorical(y_train,num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes=num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model=Sequential()
model.add(Conv2D(32,(3,3),padding="valid",activation="relu",use_bias=True,kernel_initializer="glorot_uniform"
                ,bias_initializer="zeros",strides=1,input_shape=x_train.shape[1:]))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
##normalization layer
model.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True, beta_initializer='zeros'))
model.add(Conv2D(96,(3,3),activation="relu"))
model.add(Conv2D(96,(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True, beta_initializer='zeros'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1100,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

opt=keras.optimizers.Adam(lr=0.001,decay=1e-6)
model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test,y_test))
scores=model.evaluate(x_test,y_test,verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

