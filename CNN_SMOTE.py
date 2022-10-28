# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:04:58 2022

Objective: To AUGMENT and save TESTING TRAINING AND VALIDATION DATA for CNN
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

#LOAD NPY FIES FOR DATA AND LABELS
Y=np.load('label.npy',allow_pickle=True)
X=np.load('config.npy',allow_pickle=True)

#OVERSAMPLE THE DATASET
sample_strategy= Counter(Y)#creates dict to get the classes of the label file
sample_strategy= dict.fromkeys(sample_strategy,500) #sets the number of augmented samples to 1000 #int(nnofs*1.5) 
sm = SMOTE(sampling_strategy = sample_strategy,  k_neighbors=2) # add if needed sampling_strategy = sample_strategy,
x_res, y_res = sm.fit_resample(X,Y)

#y_res=np.load('label.npy',allow_pickle=True)
#x_res=np.load('config.npy',allow_pickle=True)

#SPLIT OVERSAMPLED DATASET FOR TRAINING AND TESTING 
df = pd.DataFrame(x_res,y_res) 
training_data, testing_data = train_test_split(df, test_size=0.5)
train_x= training_data.values; train_y= np.array(training_data.index)

#SAVE TRAINING DATASET
np.save("Xa",train_x)
print("File Xa.npy has been saved")
np.save("Ya",train_y)
print("File Ya.npy has been saved")

#SPLIT THE DATASET EVENLY INTO TESTING AND VALIDATION
test_data, valid_data = train_test_split(testing_data, test_size=0.6)
test_x= test_data.values; test_y= np.array(test_data.index)
valid_x= valid_data.values; valid_y= np.array(valid_data.index)

#SAVE TESTING AND VALIDATION DATASET
np.save("Xb",test_x)
print("File Xb.npy has been saved")
np.save("Yb",test_y)
print("File Yb.npy has been saved")
np.save("Xc",valid_x)
print("File Xc.npy has been saved")
np.save("Yc",valid_y)
print("File Yc.npy has been saved")
