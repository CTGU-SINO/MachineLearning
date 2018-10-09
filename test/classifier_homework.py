# coding=UTF-8
'''
Created on 2017年6月21日
@author: Alex
'''

import numpy as np
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def readDataSet(fileName):
    fr = open(fileName)           
    lines = fr.readlines()
    print(lines[0:1])
    length = len(lines)
    dataSet = np.zeros([length,54],int)
    dataLables = np.zeros([length],int)
    for i in range(0,length):
        listNum = []
        listStr = (str(lines[i:i+1])[2:-4]).split(sep=' ')
        for j in listStr:
            listNum.append(int(j))
        dataSet[i] = listNum[0:54]
        dataLables[i] = listNum[-1]
        print(listNum)
        print(dataLables)
    return dataSet,dataLables

def text_save(content,filename,mode='a'):
    with open(filename,'a') as f:
        for i in content:
            f.write(str(i)+'\n')

#read dataSet
train_dataSet, train_hwLabels = readDataSet('data/data_train.txt')

#read  testing dataSet
dataSet,hwLabels = readDataSet('data/data_test.txt')
print(len(dataSet))

#knn
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)
res_knn = knn.predict(dataSet)
print(res_knn)

#DecisionTree
dt = DecisionTreeClassifier().fit(train_dataSet, train_hwLabels)
res_dt = dt.predict(dataSet) 
print(res_dt)

#GaussianNB
gnb = GaussianNB().fit(train_dataSet, train_hwLabels)
res_gnb = gnb.predict(dataSet)
print(res_gnb)

text_save(res_knn,'model_1.txt')
text_save(res_dt,'model_2.txt')
text_save(res_gnb,'model_3.txt')

