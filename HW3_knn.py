# -*- coding: utf-8 -*-
"""
Xurui Zhao
Class: CS677 - Summer 2
Date: 7/22/2019
Homework Problem #3
Description of Problem : 
   Implement a knn classifier
"""

import os
import numpy as np
import pandas as pd
import matplotlib . pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn . preprocessing import StandardScaler , LabelEncoder
from sklearn . neighbors import KNeighborsClassifier
from sklearn . model_selection import train_test_split

ticker='ZBRA'
input_dir = r''
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)
df['Return'] = 100.0 * df['Return'] #In percentage

def getx(y): 
    '''This function gives the mean and sd'''    
    mean=[]
    sd=[]
    index=[]
    
    dfy=df[df['Year'] == y]
    dfy = dfy.reset_index(drop = True)
    for i in range (dfy['Week_Number'][0],max(dfy['Week_Number'])+1):
        dfw = dfy[dfy['Week_Number']==i]
        dfw = dfw.reset_index(drop = True)
        index.append(i)
        mean.append(np.mean(dfw['Return']))
        sd.append(np.std(dfw['Return']))
        
        
    week = pd.DataFrame(mean,sd)
    return week

def gety(y): 
    '''This function gives the label'''
    tlabel=[]
    
    dfy=df[df['Year'] == y]
    dfy = dfy.reset_index(drop = True)
    for i in range (dfy['Week_Number'][0],max(dfy['Week_Number'])+1):
        dfw = dfy[dfy['Week_Number']==i]
        dfw = dfw.reset_index(drop = True)
        tlabel.append(dfw['Label'][0])
        
    return tlabel

X_train = getx(2017)
X_test = getx(2018)
scaler = StandardScaler ()
scaler .fit(X_train)
scaler .fit(X_test)

X_train = scaler . transform (X_train)
X_test = scaler . transform (X_test)

le = LabelEncoder ()
Y_train = le.fit_transform (gety(2017)) #red is 1, green is 0
Y_test = le.fit_transform (gety(2018))


accuracy = []
klist = []
for k in range (1 ,13):
    knn_classifier = KNeighborsClassifier ( n_neighbors =k)
    knn_classifier . fit ( X_train , Y_train )
    pred_k = knn_classifier . predict ( X_test )
    accuracy . append (np. mean ( pred_k == Y_test ))
    klist.append(k)

plt.plot(klist, accuracy)
plt . title ('Accuracy vs. k for Stock Label ')
plt . xlabel ('number of neighbors : k')
plt . ylabel ('Accuracy')

print(accuracy)
print(confusion_matrix(Y_test, pred_k))
print('')

pred=[]
for i in range(len(pred_k)):  #transfer prediction to green and red
    if pred_k[i] == 1:
        pred.append('red')
    if pred_k[i] == 0:
        pred.append('green')

dfy=df[df['Year'] == 2018] #make a test year as a new data frame
dfy = dfy.reset_index(drop = True)

plabel=[]

for w in range (dfy['Week_Number'][0],max(dfy['Week_Number'])+1):
    dfw = dfy[dfy['Week_Number']==w]
    for d in range (len(dfw)):
        plabel.append(pred[w])    #put the label on each day
                       
dfy['prediction'] = plabel

share=0
bal=100
for x in range(len(dfy)):       
  if(dfy['prediction'][x]=='green' and share==0): #buy
            share=bal/dfy['Open'][x]
            print(share)
            bal=0
  if (dfy['prediction'][x]=='red' and share!=0):   #sell             
            bal=share*dfy['Adj Close'][x-1]            
            share=0
    
print('\n"buy-and-hold" strategy: The final amount of money is ${:.2f}'. 
          format((100/dfy['Open'][0])*dfy['Adj Close'][x]))
print('Based on labels: The final amount of money is ${:.2f}'.
          format(bal+share*dfy['Adj Close'][x]) )
print('')














