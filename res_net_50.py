
#this net is the 32 layer res net architectur
##this will follow there bottleneck architecture
import keras
from keras import layers
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,AveragePooling2D,BatchNormalization,Input,Dense,Activation,Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10


##First lets build the residual bottleneck block from the res net architecture
##this uses 3 concolutional layers rather than 2 as in the standard

def residual_block(input_tensor,kernel_size,filters,stride=(1,1),double_input=False):
    """The parameters passed to this block are
        input_tensor:this is the input node that is passed as the input to the residual block
        kernal_size:this is the size of the weights like 1X1 or 3X3
        filters:these are the number of filters list that contains the filters for the there layers
        stage:this the layer at which we are...just to do the naming of the layers
        block :this is also used for the naming scheme"""
    filters1,filters2,filters3=filters
    
    ##first 1X1 conv layer
    x = BatchNormalization(axis=1, epsilon=0.001)(input_tensor)
    x = Conv2D(filters1, (1, 1),activation="relu",strides=(1,1),padding='same',kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.l2(0.01))(x)
    if double_input:
        input_tensor=Conv2D(filters1,(3,3),strides=(2,2),padding='same')(input_tensor)
    
    ##secind conv layer
    x = BatchNormalization(axis=1, epsilon=0.001)(x)
    x = Conv2D(filters2, kernel_size,
               padding='same',activation="relu",strides=stride,kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.l2(0.01))(x)
   
    ##third 1X1 conv layer
    
    x = BatchNormalization(axis=1, epsilon=0.001)(x)
    x = Conv2D(filters3, (1, 1),strides=(1,1),padding='same',kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
    

batch_size=32
num_classes=10
epochs=100
data_augmentation=False

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

##one hot excode the y_train and y_test
##the shape will be [00000010000](one hot encoded)
##just a example not the any of y_train or y_test actual resulting shape
y_train= keras.utils.to_categorical(y_train,num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes=num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


input_shape=x_train.shape[1:]

img_input = Input(shape=input_shape)
x = Conv2D(16,(3,3), strides=(1,1),padding="same",activation="relu",kernel_initializer='glorot_uniform',
               kernel_regularizer=keras.regularizers.l2(0.01))(img_input)
x = BatchNormalization(axis=1)(x)

x = residual_block(x, 3, [16, 16, 16],(1,1))
x = residual_block(x, 3, [16, 16, 16],(1,1))
x = residual_block(x, 3, [16, 16, 16],(1,1))
x = residual_block(x, 3, [32, 32, 32],(2,2),True)
x = residual_block(x, 3, [32, 32, 32],(1,1))
x = residual_block(x, 3, [32, 32, 32],(1,1))
x = residual_block(x, 3, [32, 32, 32],(1,1))
x = residual_block(x, 3, [64, 64, 64],(2,2),True)
x = residual_block(x, 3, [64, 64, 64],(1,1))
x = residual_block(x, 3, [64, 64, 64],(1,1))
x = residual_block(x, 3, [64, 64, 64],(1,1))
x = residual_block(x, 3, [64, 64, 64],(1,1))
x = residual_block(x, 3, [64, 64, 64],(1,1))
x = residual_block(x, 3, [128, 128, 128],(2,2),True)
x = residual_block(x, 3, [128, 128, 128],(1,1))
x = residual_block(x, 3, [128, 128, 128],(1,1))

##averagePool
x = AveragePooling2D((2,2), name='avg_pool')(x)
x = Flatten()(x)
x = Dense(num_classes, activation='softmax')(x)

inputs = img_input
model = Model(inputs, x)
opt=keras.optimizers.Adam(lr=0.001,decay=1e-6)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])


if not data_augmentation:
    model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test,y_test))
else:
     datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

     # Compute quantities required for feature-wise normalization
     # (std, mean, and principal components if ZCA whitening is applied).
     datagen.fit(x_train)

     # Fit the model on the batches generated by datagen.flow().
     model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4,
                        steps_per_epoch=50000)


scores=model.evaluate(x_test,y_test,verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

