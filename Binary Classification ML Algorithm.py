"""
A binary classification machine learning algorithm for identification of normal walking and walking with a handicap 

This Script was created for the course Computational Intelligence in Engineering hold by Arnd Koeppe and Marion Mundt, supervised by Prof. Bernd Markert @ Institue of General Mechanics RWTH Aachen University 

Date: 20.11.2019

Authors (Github Profile): Ahmet Küpeli (KuepeliAhmet), Vivek Chavan (Vivek9Chavan), Chi Shing Li (@charles4444)

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import gc
gc.collect()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import glob
from scipy.signal import find_peaks
import scipy.special
from math import e,sqrt,sin,cos

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Libraries for cross validation
from sklearn.model_selection import KFold


global_max_x=0
global_min_x=1000
global_max_y=0
global_min_y=1000
global_max_z=0
global_min_z=1000
files=[]

Total =glob.glob('C:/')

Subject=[]
counter=-1
for i in range(1,22):
    i_str=str(i).zfill(2)
    Subject.append(glob.glob('C:*' + str(i_str) + '_' + '*'))
    

for i in range(0,len(Subject)):
    for j in range(0,len(Subject[i])):
        counter=counter+1
        Subject[i][j]= counter

#Variables
Accelerometer=[]
data=[]
Time=[]
acc=[]
acc_pca=[]
Acceleration_x=[]
Acceleration_y=[]
Acceleration_z=[]

dtime=[0]*len(Total)
mod_x= [0]*len(Total)
mod_y= [0]*len(Total)
mod_z= [0]*len(Total)
Delta_t=[0]*len(Total)
impaired=[]
Tim_All=[0]*len(Total)

for i in range(0,len(Total)):
    
    ### NEW - Checking if normal or impaired- if normal bool=0, impaired=1
    if 'normal' in Total[i]:
        impaired.append(0)
    elif 'impaired' in Total[i]:
        impaired.append(1)

    Accelerometer.append(Total[i] + '/Accelerometer.csv')
    data.append(pd.read_csv(Accelerometer[i], sep=",")) 
    # Time[x][y]: x= Filenumber, y= Length of Time
    Time.append(data[i].iloc[0:,0])
    acc.append(data[i].iloc[0:,1:4])
    
    #### Rotating Data
    def rotate (x, theta, axis='x'):
        c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
        if axis == 'x': return np.dot(x, np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
            ]))
        elif axis == 'y': return np.dot(x, np.array([
            [c, 0, -s],
            [0, 1,  0],
            [s, 0,  c]
            ]))
        elif axis == 'z': return np.dot(x, np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
            ]))

    # PCA, cov 3x3 Matrix = A.T * A
    cov = np.matmul(acc[i].values.astype(float).T,acc[i].values.astype(float))
    # PCA principal component analysis, cov: mxn dimensions -> m=3, n=3
    # u = mxm unitary matric (ex, 2x2,3x3,4x4,...)
    # s = singular value, here as vector, mathematically its a diagonal matrix (s1,0,0 ; 0, s2 ,0 ;...)
    # v = conjugate transpose matrix nxn, here 3x3 transponend
    u, s, v = np.linalg.svd(cov)
    acc_pca.append(np.dot(acc[i].values.astype(float),u))
    
    # We are checking mean Acc_x because x is the only mean number that will be 
    # clearly above or under zero (~+10, -10), Rotating z, because z-stays constant acc. to 3 finger rule
    if np.mean(acc_pca[i][:,0]) < 0:
        acc_pca[i] = rotate(acc_pca[i], 180, axis='z')

    #Applying Butterworth Filter, len(Acceleration_X) stays the same, (magnitude of order, sampling)
    fs = 1/Time[i][1] # Sampling frequency
    dt=1/fs
    dtime[i]=dt
    fc = 5  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Window Length
    # print('w = %f' %w)
    b, a = signal.butter(5, w, 'low')
    acc_pca[i]=signal.filtfilt(b, a, acc_pca[i], axis=0) #overwriting acc_pca with filtered data
    Acceleration_x.append(acc_pca[i][0:,0])
    Acceleration_y.append(acc_pca[i][0:,1])
    Acceleration_z.append(acc_pca[i][0:,2])

    # peak,_ = just saves the index position within the list, not the actual value
    # Need to find Alternative to Height!
    peak_x,_= find_peaks(Acceleration_x[i], distance=100, prominence=3)
    peak_y,_= find_peaks(Acceleration_y[i], distance=100, prominence=3)
    peak_z,_=find_peaks(Acceleration_z[i], distance=100, prominence=3)
    
    # Variables for Tuple-List Transformation, Peak finding
    index_peak_x=[] # Index Positions, where peaks are found
    index_peak_y=[]
    index_peak_z=[]
        
    # Transforming Tuple into List
    for x in range(0, len(peak_x)):
        index_peak_x.append(peak_x[x])  #X-Acc
    
    for x in range(0, len(peak_y)):
        index_peak_y.append(peak_y[x]) #Y-Acc
    
    for x in range(0, len(peak_z)):
        index_peak_z.append(peak_z[x]) #Z-Acc

    # CUT-OUT THE REVELANT SEGMENT
    Tim=[]
    Tim1=[]
    h_tot=[]
    
    for x in index_peak_x:
        Tim.append(Time[i][x])
        h_tot.append(Acceleration_x[i][x])

    seg_avg=(Tim[-1]-Tim[0])/(len(Tim)-1)
    
    h_avg=sum(h_tot)/len(index_peak_x)
    
    Tim1_a=0
    Tim1_b=Tim[-1]

    Tim1_a=Tim[3]
    Tim1_b=Tim[-2]
    
    if Tim1_b-Tim1_a>25:
        for x in range(1,20):
            if abs(Tim[x]-Tim[x-1])<seg_avg:
                if abs(h_tot[x]-h_tot[x-1])<0.4*h_avg:
                    x=x+2
                    Tim1_a=Tim[x+4]
            else:
                x=x+2
                Tim1_a=Tim[x+4]
                break
        
        for x in range(1,(len(Tim)-2)):
            if abs(Tim[x]-Tim[x+1])>seg_avg:
                continue
                Tim1_b= Tim[x-1]
                break
            else:
                x=x-1
                Tim1_b= Tim[x-1]
                

    Delta_t[i]=np.arange(Tim1_a,Tim1_b, dt)
    
    Tim_All[i]=(Tim1_a, Tim1_b)
    
    # Modified Value
    mod_x[i] = np.interp(Delta_t[i], Time[i],Acceleration_x[i])
    mod_y[i] = np.interp(Delta_t[i], Time[i],Acceleration_y[i])
    mod_z[i] = np.interp(Delta_t[i], Time[i],Acceleration_z[i])
    
    lokal_max_x=np.max(mod_x[i])
    lokal_min_x=np.min(mod_x[i])
    lokal_max_y=np.max(mod_y[i])
    lokal_min_y=np.min(mod_y[i])
    lokal_max_z=np.max(mod_z[i])
    lokal_min_z=np.min(mod_z[i])
    
    if lokal_max_x > global_max_x:
        global_max_x=lokal_max_x
    
    if lokal_min_x < global_min_x:
        global_min_x=lokal_min_x

    
    if lokal_max_y > global_max_y:
        global_max_y=lokal_max_y
    
    if lokal_min_y < global_min_y:
        global_min_y=lokal_min_y
        
    if lokal_max_z > global_max_z:
        global_max_z=lokal_max_z
    
    if lokal_min_z < global_min_z:
        global_min_z=lokal_min_z


x_NORM=[0]*len(Total)
y_NORM=[0]*len(Total)
z_NORM=[0]*len(Total)
steps=[]
steps_new=[]

for i in range(0,len(Total)):
    steps_loop_new=[]
    steps_loop=[]
    step_length=[]
    seq_sam_x=[]
    seq_sam_y=[]
    seq_sam_z=[]
    seq_sam=[]
    sam_x_new=[]
    sam_y_new=[]
    sam_z_new=[]
    #for x in range(0,(len(mod_x[i]))):
    x_NORM[i]=(mod_x[i]-global_min_x)/(global_max_x-global_min_x)
    y_NORM[i]=(mod_y[i]-global_min_y)/(global_max_y-global_min_y)
    z_NORM[i]=(mod_z[i]-global_min_z)/(global_max_z-global_min_z)
    
    dist= 0.7/dtime[i]

    peak_x_new,_= find_peaks(x_NORM[i], distance=dist, height=0.35)
    peak_y_new,_= find_peaks(y_NORM[i], distance=dist, height=0.35)
    peak_z_new,_= find_peaks(z_NORM[i], distance=dist, height=0.35)
    
    for x in range(0,len(peak_x_new)-1):
        step_length.append(peak_x_new[x+1]-peak_x_new[x])
    
    ### Segmentating x_NORM into each step_length   
    seq_x = ([x_NORM[i][sum(step_length[:k]):sum(step_length[:k])+n] for k,n in enumerate(step_length)])
    seq_y = ([y_NORM[i][sum(step_length[:k]):sum(step_length[:k])+n] for k,n in enumerate(step_length)])
    seq_z = ([z_NORM[i][sum(step_length[:k]):sum(step_length[:k])+n] for k,n in enumerate(step_length)])
    

    ### Resampling
    for x in range(0,len(step_length)):
        seq_sam_x.append((signal.resample(seq_x[x],101)))
        seq_sam_y.append((signal.resample(seq_y[x],101)))
        seq_sam_z.append((signal.resample(seq_z[x],101)))
        steps_loop.append(np.vstack((seq_sam_x[x],seq_sam_y[x],seq_sam_z[x])).T)
    steps.append(steps_loop)

Subj_tmp=Subject
Subject=[]
# Subject folder, including all steps that a Subject has taken over his 6 Walks
for i in range(0,len(Subj_tmp)):
    Subject_steps=[]
    for j in range(0,len(Subj_tmp[i])):
        FileNr_tmp=Subj_tmp[i][j]
        #File_steps_tmp=[]
        #File_steps_tmp=File_steps_tmp + steps[FileNr_tmp]
        Subject_steps=Subject_steps + steps[FileNr_tmp]
    Subject.append(Subject_steps)

# Creating a list with impaired data for new Subject[] list
impaired_steps=[]
for i in range(0,len(Subject)):
    impaired_steps_subj=[]
    for j in range(0,len(Subj_tmp[i])):
        impaired_steps_folder=[]
        FileNr_tmp=Subj_tmp[i][j]
        if impaired[FileNr_tmp] == 1:
            impaired_steps_folder=[1]*len(steps[FileNr_tmp])
        elif impaired[FileNr_tmp] == 0:
            impaired_steps_folder=[0]*len(steps[FileNr_tmp])
        
        impaired_steps_subj=impaired_steps_subj + impaired_steps_folder
        
    impaired_steps.append(impaired_steps_subj)

accuracies=[]
losses=[]
n_fold=10
kf = KFold(n_fold)
#neurons=[25,50,75,100,125,150,303]
#epochs=[1,50]
#results_acc=[]
#results_loss=[]

for train_index, test_index in kf.split(Subject):
    print("TRAIN:", train_index, "TEST:", test_index)
    
    # train steps and test steps
    train_step=[]
    test_step=[]
    impaired_steps_train=[]
    impaired_steps_test=[]
    
    for i in range(0,len(train_index)):
        train_step=train_step+ Subject[train_index[i]]
        impaired_steps_train=impaired_steps_train + impaired_steps[train_index[i]]
    for i in range(0,len(test_index)):
        test_step=test_step+ Subject[test_index[i]]
        impaired_steps_test=impaired_steps_test + impaired_steps[test_index[i]]
    
        
    ### Changing Dimensions for Tensorflow
    train_step = np.asarray(train_step, dtype=np.float32)
    test_step = np.asarray(test_step, dtype=np.float32)

    impaired_steps_train=np.asarray(impaired_steps_train, dtype=np.uint8)
    impaired_steps_test=np.asarray(impaired_steps_test, dtype=np.uint8)
    
    
    ###############################################
    ##              NEURAL NETWORK
    #################################################
    
    # For Loop see:https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    
    # Creating the Model
    # Tutorial Keras:https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    # model = keras.Sequential([keras.layers.Flatten(input_shape=(101,3)), keras.layers.Dense(90, activation='relu'), keras.layers.Dense(90, activation='sigmoid'), keras.layers.Dense(90, activation='tanh'), keras.layers.Dense(2, activation='softmax')])
    model = keras.Sequential([keras.layers.Flatten(input_shape=(101, 3)), keras.layers.Dense(100, activation='relu'), keras.layers.Dense(2, activation='sigmoid')])
       
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #Training Model with train data
    #Validation split 0.22, because we want 20% out of 90%, 20/90=0,22
    model.fit(train_step, impaired_steps_train, epochs=300 ,shuffle=True);
    
    #Evaluating created model with test data
    loss, accuracy = model.evaluate(test_step, impaired_steps_test, verbose=2);
    
    print('Loss=', loss)
    print('accuracy=', accuracy)

    accuracies.append(accuracy)
    losses.append(loss)
    
    #################################################
    ##              MAKE PREDICTIONS
    #################################################
    
    # Show the predictions for the test set
    predictions = model.predict(test_step)
    #print(np.round(predictions))
    
    
print(losses)
print(accuracies)
print('average accuracy for  Fold CV : ' + str(np.mean(accuracies)))


    

    
