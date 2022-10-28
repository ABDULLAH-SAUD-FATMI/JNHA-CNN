"""
Name: mycnn_model

Objective: to create and save a CNN model for the JNHA dataset
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, BatchNormalization
from keras.utils import np_utils
from keras import callbacks
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

## import data
y_train=np.load('Ya.npy',allow_pickle=True)
X_train=np.load('Xa.npy',allow_pickle=True)
y_test=np.load('Yb.npy',allow_pickle=True)
X_test=np.load('Xb.npy',allow_pickle=True)

# encode class values as integers
#encoded_y=
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
Y_train = np_utils.to_categorical(encoded_Y, 4)
encoder.fit(y_test)
encoded_y = encoder.transform(y_test)
Y_test = np_utils.to_categorical(encoded_y, 4)

dim=Y_train[0].shape[0]
nofcolumn=X_train.shape[1]
n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], Y_train.shape[0]
neuronwidth=150;
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1) #reshape training data as convolution layers only word with 3d arrays
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
#create model
model = Sequential()
#ADDING CONVOLUTION LAYER
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=4))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=4))
#model.add(BatchNormalization())

# flatten output of conv
model.add(Flatten())
model.add(Dense(neuronwidth, input_dim=nofcolumn, activation='relu'))
model.add(Dense(dim, activation='softmax'))

# Compile model
model.compile(optimizer ="adam", loss ="categorical_crossentropy", metrics =['accuracy'])

# early stopping to prevent overfitting of model
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",       
                                        mode ="min", patience = 20, min_delta=0.01, 
                                        restore_best_weights = True)
#fitting model
history = model.fit(X_train, Y_train, batch_size =2, 
                    epochs = 300, validation_data =(X_test, Y_test), 
                    callbacks =[earlystopping]
                    )

#PLOTTING LEARNING CURVE
run_no=int(len(os.listdir("learning_curves\\")))+1 # count for saving learning curve name
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,len(history.history['loss']))
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.rcParams["figure.figsize"] = (16,10)
plt.legend()
plt.savefig(os.getcwd()+'/learning_curves/train_val_loss#'+str(run_no)+'.png',dpi=400,bbox_inches='tight')
plt.show()

#saving model
model.save("model.h5")
