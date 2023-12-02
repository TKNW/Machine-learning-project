import pandas as pd
from ast import literal_eval
import numpy as np
import random as rd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
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
dataset_test=pd.read_csv("Data\\test.csv")
print("Read file succeed.")

count=0
locate_from=[]
locate_to=[]
for i in range(len(dataset_test['POLYLINE'])):
    f=dataset_test['POLYLINE'][i]
    f=literal_eval(f)
    print(f"{i}")
    if f:
        print(f"{f[0]},{f[-1]}")
        locate_from.append(f[0])
        locate_to.append(f[-1])
    else:
        count=count+1
        print(f"{i} is empty")
        dataset_test.drop([i],axis=0,inplace=True)
dataset_test['from']=locate_from
dataset_test['to']=locate_to

vectors = []
for i in range(len(dataset_test['POLYLINE'])):

    f = dataset_test['POLYLINE'][i]

    f = literal_eval(f)

    print(f"{i}")

    vector = []

    for j in range(1, len(f)):

        x = np.array(f[j])-np.array(f[j-1])

        vector.append(x)

    vectors.append(vector)

dataset_test["vectors"] = vectors


dataset_test.drop("DAY_TYPE",axis=1,inplace=True)
dataset_test.drop("POLYLINE",axis=1,inplace=True)

dataset_test.drop("TAXI_ID",axis=1,inplace=True)

import datetime

# DECODING TIME SIGNATURE TEST DATA
dataset_test["TIMESTAMP"] = [float(time) for time in dataset_test["TIMESTAMP"]]
dataset_test["data_time"] = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in dataset_test["TIMESTAMP"]]
dataset_test

for i in range(0,len(dataset_test["vectors"])):
  if len(dataset_test["vectors"][i])<5:
    print(f'第{i}列不足5,僅有{len(dataset_test["vectors"][i])}個元素')
    for x in range(0,5-len(dataset_test["vectors"][i])):
      dataset_test["vectors"][i].insert(-1,[0,0])
dataset_test

dataset_test.drop("TIMESTAMP",axis=1,inplace=True)
dataset_test

dataset_test["Year"] = dataset_test["data_time"].dt.year
dataset_test["Month"] = dataset_test["data_time"].dt.month
dataset_test["Week"] = dataset_test["data_time"].dt.isocalendar().week
dataset_test["Day"] = dataset_test["data_time"].dt.day
dataset_test["Hour"] = dataset_test["data_time"].dt.hour
dataset_test["Min"] = dataset_test["data_time"].dt.minute
dataset_test["Weekday"] =dataset_test["data_time"].dt.weekday
dataset_test

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset_test["MISSING_DATA"]= le.fit_transform(dataset_test["MISSING_DATA"])
x_from=[]
y_from=[]
for i in range(len(dataset_test['from'])):
  x=dataset_test['from'][i][0]
  y=dataset_test['from'][i][1]
  x_from.append(x)
  y_from.append(y)
dataset_test["x_from"]=x_from
dataset_test["y_from"]=y_from
x_to=[]
y_to=[]
for i in range(len(dataset_test['to'])):
  x=dataset_test['to'][i][0]
  y=dataset_test['to'][i][1]
  x_to.append(x)
  y_to.append(y)
dataset_test["x_to"]=x_to
dataset_test["y_to"]=y_to
dataset_test

encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(dataset_test[['CALL_TYPE']]).toarray())
dataset_test = dataset_test.join(encoder_df)
dataset_test.rename(columns={0:'call_type_a', 1:'call_type_b',2:'call_type_c'}, inplace=True)

dataset_test['call_type_a']= [int(k) for k in dataset_test["call_type_a"]]
dataset_test['call_type_b'] =[int(k) for k in dataset_test["call_type_b"]]
dataset_test['call_type_c']= [int(k) for k in dataset_test["call_type_c"]]

from tqdm import tqdm
x_1=[]
x_2=[]
x_3=[]
x_4=[]
x_5=[]
y_1=[]
y_2=[]
y_3=[]
y_4=[]
y_5=[]
pick=5
for i in tqdm(range(len(dataset_test['vectors']))):
  j=dataset_test['vectors'][i]
  if len(j)<pick:
    dataset_test.drop([i],axis=0,inplace=True)
    continue
  L1=rd.sample(range(0,(len(dataset_test['vectors'][i]))),pick)
  L1=sorted(L1)
  x1=j[L1[0]][0]
  y1=j[L1[0]][1]
  x2=j[L1[1]][0]
  y2=j[L1[1]][1]
  x3=j[L1[2]][0]
  y3=j[L1[2]][1]
  x4=j[L1[3]][0]
  y4=j[L1[3]][1]
  x5=j[L1[4]][0]
  y5=j[L1[4]][1]
  x_1.append(x1)
  y_1.append(y1)
  x_2.append(x2)
  y_2.append(y2)
  x_3.append(x3)
  y_3.append(y3)
  x_4.append(x4)
  y_4.append(y4)
  x_5.append(x5)
  y_5.append(y5)
dataset_test["x_1"]=x_1
dataset_test["y_1"]=y_1
dataset_test["x_2"]=x_2
dataset_test["y_2"]=y_2
dataset_test["x_3"]=x_3
dataset_test["y_3"]=y_3
dataset_test["x_4"]=x_4
dataset_test["y_4"]=y_4
dataset_test["x_5"]=x_5
dataset_test["y_5"]=y_5
dataset_test.drop("from",axis=1,inplace=True)
dataset_test.drop("to",axis=1,inplace=True)
dataset_test.drop("vectors",axis=1,inplace=True)

dataset_test.drop("data_time",axis=1,inplace=True)

tenp_column=dataset_test.pop('x_to')
dataset_test.insert(len(dataset_test.columns),"x_to",tenp_column)

tenp_column=dataset_test.pop('y_to')
dataset_test.insert(len(dataset_test.columns),"y_to",tenp_column)


chunks = np.array_split(dataset_test.index, 100) # split into 100 chunks
for chunck, subset in enumerate(tqdm(chunks)):
    if chunck == 0: # first row
        dataset_test.loc[subset].to_csv('Data\\test_our.csv', mode='w', index=True)
    else:
        dataset_test.loc[subset].to_csv('Data\\test_our.csv', header=None, mode='a', index=True)
#print(dataset["Data_time"])