import pandas as pd
from ast import literal_eval
import numpy as np
import random as rd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
import datetime

def vector_to_list(old_str):
  new_str=old_str.replace("array(","")
  new_str=new_str.replace(")","")
  new_str=literal_eval(new_str)
  return new_str

if len(sys.argv) < 2:
   print("Need pick number.")
   sys.exit()
dataset=pd.read_csv("new_train2_test.csv")
print("Read file succeed.")
dataset.drop("Unnamed: 0",axis=1,inplace=True)
dataset.drop("Unnamed: 0.1",axis=1,inplace=True)
tenp_column=dataset.pop('to')
dataset.insert(3,"to",tenp_column)

dataset["TIMESTAMP"] = [float(time) for time in dataset["TIMESTAMP"]]
dataset["Data_time"] = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in dataset["TIMESTAMP"]]
dataset["Year"] = dataset["Data_time"].dt.year
dataset["Month"] = dataset["Data_time"].dt.month
dataset["Week"] = dataset["Data_time"].dt.isocalendar().week
dataset["Day"] = dataset["Data_time"].dt.day
dataset["Hour"] = dataset["Data_time"].dt.hour
dataset["Min"] = dataset["Data_time"].dt.minute
dataset["Weekday"] = dataset["Data_time"].dt.weekday
print("Transfer date finish.")

le = LabelEncoder()
dataset["MISSING_DATA"]= le.fit_transform(dataset["MISSING_DATA"])
dataset['from']=dataset['from'].apply(literal_eval)
dataset['to']=dataset['to'].apply(literal_eval)
x_from=[]
y_from=[]
x_to=[]
y_to=[]
for i in range(len(dataset['from'])):
  x_from.append(dataset['from'][i][0])
  y_from.append(dataset['from'][i][1])
dataset["x_from"]=x_from
dataset["y_from"]=y_from
for i in range(len(dataset['to'])):
  x_to.append(dataset['to'][i][0])
  y_to.append(dataset['to'][i][1])
dataset["x_to"]=x_to
dataset["y_to"]=y_to

pick = int(sys.argv[1])
vectorlist=[[] for _ in range(pick*2)]
index = 0
for i in tqdm(range(len(dataset['vectors']))):
  j=dataset['vectors'][i]
  j=vector_to_list(j)
  if len(j)<pick+1:
    dataset.drop([i],axis=0,inplace=True)
    continue
  L1=rd.sample(range(0,len(j)-1),pick)
  L1=sorted(L1)
  for k in range(0, pick):
    index = k * 2
    vectorlist[index].append(j[L1[k]][0])
    vectorlist[index+1].append(j[L1[k]][1])
for i in range(0 , pick):
  index = i * 2
  dataset["x_" + str(i+1)] = vectorlist[index]
  dataset["y_" + str(i+1)] = vectorlist[index+1]

print("Random select finish.")
dataset.drop("from",axis=1,inplace=True)
dataset.drop("to",axis=1,inplace=True)
dataset.drop("vectors",axis=1,inplace=True)
chunks = np.array_split(dataset.index, 100) # split into 100 chunks
for chunck, subset in enumerate(tqdm(chunks)):
    if chunck == 0: # first row
        dataset.loc[subset].to_csv('data_3.csv', mode='w', index=True)
    else:
        dataset.loc[subset].to_csv('data_3.csv', header=None, mode='a', index=True)
#print(dataset["Data_time"])