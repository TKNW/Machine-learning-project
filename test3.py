import pandas as pd
import ast
from ast import literal_eval
import numpy as np
import random as rd
from tqdm import tqdm
import csv
dataset=pd.read_csv("new_train2.csv")
dataset.drop("Unnamed: 0",axis=1,inplace=True)
dataset.drop("Unnamed: 0.1",axis=1,inplace=True)
tenp_column=dataset.pop('to')
dataset.insert(3,"to",tenp_column)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset["MISSING_DATA"]= le.fit_transform(dataset["MISSING_DATA"])
dataset['from']=dataset['from'].apply(literal_eval)
dataset['to']=dataset['to'].apply(literal_eval)
x_from=[]
y_from=[]
for i in range(len(dataset['from'])):
  x=dataset['from'][i][0]
  y=dataset['from'][i][1]
  x_from.append(x)
  y_from.append(y)
dataset["x_from"]=x_from
dataset["y_from"]=y_from
x_to=[]
y_to=[]
for i in range(len(dataset['to'])):
  x=dataset['to'][i][0]
  y=dataset['to'][i][1]
  x_to.append(x)
  y_to.append(y)
dataset["x_to"]=x_to
dataset["y_to"]=y_to


def vector_to_list(old_str):
  new_str=old_str.replace("array(","")
  new_str=new_str.replace(")","")
  new_str=literal_eval(new_str)
  return new_str

x_1=[]
x_2=[]
x_3=[]
y_1=[]
y_2=[]
y_3=[]
pick=3
for i in tqdm(range(len(dataset['vectors']))):
  j=dataset['vectors'][i]
  j=vector_to_list(j)
  if len(j)<pick+1:
    dataset.drop([i],axis=0,inplace=True)
    continue
  L1=rd.sample(range(0,len(j)-1),pick)
  L1=sorted(L1)
  x1=j[L1[0]][0]
  y1=j[L1[0]][1]
  x2=j[L1[1]][0]
  y2=j[L1[1]][1]
  x3=j[L1[2]][0]
  y3=j[L1[2]][1]
  x_1.append(x1)
  y_1.append(y1)
  x_2.append(x2)
  y_2.append(y2)
  x_3.append(x3)
  y_3.append(y3)
dataset["x_1"]=x_1
dataset["y_1"]=y_1
dataset["x_2"]=x_2
dataset["y_2"]=y_2
dataset["x_3"]=x_3
dataset["y_3"]=y_3
dataset.drop("from",axis=1,inplace=True)
dataset.drop("to",axis=1,inplace=True)
dataset.drop("vectors",axis=1,inplace=True)
chunks = np.array_split(dataset.index, 100) # split into 100 chunks
for chunck, subset in enumerate(tqdm(chunks)):
    if chunck == 0: # first row
        dataset.loc[subset].to_csv('data_3.csv', mode='w', index=True)
    else:
        dataset.loc[subset].to_csv('data_3.csv', header=None, mode='a', index=True)
print(dataset)
