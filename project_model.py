#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:37:47 2023

@author: python
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df = pd.read_csv('Admission_Predict.csv')
hed = df.head()
print(hed)
x = df.iloc[:, [1,2,3,4,5,6]]
y = df.iloc[:, 7]
# Train and test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
# Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# Model
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# find accuricy
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
# Confusion Matriex
cm = metrics.confusion_matrix(y_test, y_pred)
print(y)
sbn.heatmap(cm, annot=True, cmap='Blues')
plt.show()
data1 = [[337, 118, 4, 4.5, 4.5, 9.65]]
data1 = sc.transform(data1)
data2 = [[314, 103, 2, 2, 3, 3.2]]
data2 = sc.transform(data2)
res1 = model.predict(data1)
res2 = model.predict(data2)
print(res1)
print(res2)
# 1 = GRE Score 2=TOEFL Score 3=University Rating 4=SOP 5=LOR 6=CGPA
gre_score = int(input("Enter Your GRE Score : "))
toefl_score = int(input("Enter Your TOEFL score : "))
uni_rank = int(input("Enter Your University Rating : "))
sop = float(input("Enter Your SOP : "))
lor = float(input("Enter Your LOR score : "))
cgpa = float(input("Enter Your CGPA : "))
data = [[gre_score, toefl_score, uni_rank, sop, lor, cgpa]]
sc_data = sc.transform(data)
ans = model.predict(sc_data)
if ans == 0:
    print("Sorry You Are Not Eligible ...:( ")
else:
    print('Are You Eligible .... :) ')