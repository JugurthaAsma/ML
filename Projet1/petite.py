import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kppv import kppv
from scipy.io import loadmat
import os
import matplotlib.image as mpimg

def K1ppv(x,appren,oracle,K,M):
    m2 = appren.shape[1]
    d=np.zeros(m2)
    for k in range(0,m2):
        vdif=x-appren[:,k]
        d[k]=np.dot(np.dot(np.transpose(vdif),M),vdif)
    d_ind=np.argsort(d)
    vote=np.zeros(int(np.max(oracle))+1)
   #max(oracle) permet de savoir le nombre de classe
    for k in range(0,K):
        #print(k)
        #print(oracle[d_ind[k]])
        vote[int(oracle[0,d_ind[k]])]=vote[int(oracle[0,d_ind[k]])]+1
    clas=np.argmax(vote)
    return clas

def kppv(g,classg,K,x):
    M = np.array([[1,0],[0,1]])
    n = x.shape[1]
    clas=np.zeros((1,n))
    for b in range(0,n):
        clas[0,b]=K1ppv(x[:,b],g,classg,K,M)
    return clas


os.chdir("./")
X = (pd.read_excel("./WangSignatures.xls",sheet_name=0,index_col = 0,header=None))
Mesure=X.values
print(Mesure)

X = (pd.read_excel("./WangSignatures.xls",sheet_name=1,index_col = 0,header=None))
Data=X.values
Mesure=np.concatenate((Mesure,Data),axis=1)
print(Mesure)

X = (pd.read_excel("./WangSignatures.xls",sheet_name=2,index_col = 0,header=None))
Data=X.values
Mesure=np.concatenate((Mesure,Data),axis=1)
print(Mesure)
X = (pd.read_excel("./WangSignatures.xls",sheet_name=3,index_col = 0,header=None))
Data=X.values
Mesure=np.concatenate((Mesure,Data),axis=1)
print(Mesure)
X = (pd.read_excel("./WangSignatures.xls",sheet_name=4,index_col = 0,header=None))
Data=X.values
Mesure=np.concatenate((Mesure,Data),axis=1)


#nombre d'observations 
n = Mesure.shape[0] 
 
#nombre de variables 
p = Mesure.shape[1] 
 
print(X.index)
print(n)
print(p)

Label=np.zeros(1000,'int')

for b in range(0,1000):
    nom= X.index[b]
    Label[b]= int(int(nom[0:-4])/100)
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5) 
# Fit the classifier to the data
knn.fit(Mesure,Label)
res=(knn.kneighbors([Mesure[500,]],return_distance=False))

os.chdir("./Wang")
for b in range(0,5):
    nom= X.index[res[0,b]]
    img=mpimg.imread(nom)
    plt.imshow(img)
    plt.show()
    
######################################
    #Discrimination

Label=np.zeros(1000,'int')
print(Label)

for b in range(0,1000):
    nom= X.index[b]
    print(nom)
    Label[b]= int(int(nom[0:-4])/100)
    print(Label[b])

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#split dataset into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(Mesure, Label, test_size=0.2, random_state=1, stratify=Label)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5) 
# Fit the classifier to the data
knn.fit(X_train,Y_train)
rep_knn = knn.predict(X_test)

#Create a Gaussian Classifier
bayes = GaussianNB()
# Train the model using the training sets
bayes.fit(X_train,Y_train)
#Predict Output
rep_bayes= bayes.predict(X_test)

conf_bayes=confusion_matrix(Y_test,rep_bayes)
conf_knn=confusion_matrix(Y_test,rep_knn)

Pe_bayes=(1-np.trace(conf_bayes)/len(Y_test))
Pe_knn=(1-np.trace(conf_knn)/len(Y_test))

print('Bayes :',Pe_bayes)
print(conf_bayes)

print('Knn :',Pe_knn)
print(conf_knn)




import os
os.chdir('./')


def affiche_classe(x,clas,k):
    for k in range(0,k):
        
        ind=(clas==k)
        
        plt.plot(x[0,ind[0]],x[1,ind[0]],"o")
    plt.show()



#######################################
# Main

if __name__ == '__main__': 

    
    Data = loadmat('../p1_test.mat')
    test = Data['test']
    x=Data['x']
    clasapp=Data['clasapp']-1

    myclass=np.concatenate((np.zeros(shape=(1,50)),np.zeros(shape=(1,50))+1,np.zeros(shape=(1,50))+2),axis=1)
    
    affiche_classe(test,myclass,3)
    
    
    reskppv= kppv(test,myclass,3,x)

    Erreurkppv=reskppv.size - np.sum((reskppv==clasapp))
    print(Erreurkppv)
    print(Erreurkppv/reskppv.size)
   
# -*- coding: utf-8 -*-


