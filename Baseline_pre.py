import pandas as pd
import datetime
import re
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv(("Data\\train.csv"))
test = pd.read_csv(("Data\\test.csv"))

train = train.drop("DAY_TYPE", axis=1)
test = test.drop("DAY_TYPE", axis=1)

train["TIMESTAMP"] = [float(time) for time in train["TIMESTAMP"]]
train["Data_time"] = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in train["TIMESTAMP"]]
train["Year"] = train["Data_time"].dt.year
train["Month"] = train["Data_time"].dt.month
train["Week"] = train["Data_time"].dt.isocalendar().week
train["Day"] = train["Data_time"].dt.day
train["Hour"] = train["Data_time"].dt.hour
train["Min"] = train["Data_time"].dt.minute
train["Weekday"] = train["Data_time"].dt.weekday

test["TIMESTAMP"] = [float(time) for time in test["TIMESTAMP"]]
test["Data_time"] = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in test["TIMESTAMP"]]
test["Year"] = test["Data_time"].dt.year
test["Month"] = test["Data_time"].dt.month
test["Week"] = test["Data_time"].dt.isocalendar().week
test["Day"] = test["Data_time"].dt.day
test["Hour"] = test["Data_time"].dt.hour
test["Min"] = test["Data_time"].dt.minute
test["Weekday"] = test["Data_time"].dt.weekday

encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(train[['CALL_TYPE']]).toarray())
final_train = train.join(encoder_df)
final_train.rename(columns={0:'call_type_a', 1:'call_type_b',2:'call_type_c'}, inplace=True)

encoder_df = pd.DataFrame(encoder.fit_transform(test[['CALL_TYPE']]).toarray())
final_test = test.join(encoder_df)
final_test.rename(columns={0:'call_type_a', 1:'call_type_b',2:'call_type_c'}, inplace=True)

lists_1st_lon = []
for i in range(0,len(final_train["POLYLINE"])):
    if final_train["POLYLINE"][i] == '[]':
        k=0
        lists_1st_lon.append(k)
    else:
        k = re.sub(r"[[|[|]|]|]]", "", final_train["POLYLINE"][i]).split(",")[0]
        lists_1st_lon.append(k)
        
final_train["lon_1st"] = lists_1st_lon

# 1st lat
lists_1st_lat = []
for i in range(0,len(final_train["POLYLINE"])):
    if final_train["POLYLINE"][i] == '[]':
        k=0
        lists_1st_lat.append(k)
    else:
        k = re.sub(r"[[|[|]|]|]]", "", final_train["POLYLINE"][i]).split(",")[1]
        lists_1st_lat.append(k)
        
final_train["lat_1st"] = lists_1st_lat

lists_last_lon = []
for i in range(0,len(final_train["POLYLINE"])):
        if final_train["POLYLINE"][i] == '[]':
            k=0
            lists_last_lon.append(k)
        else:
            k = re.sub(r"[[|[|]|]|]]", "", final_train["POLYLINE"][i]).split(",")[-2]
            lists_last_lon.append(k)

final_train["lon_last"] = lists_last_lon

# last lat
lists_last_lat = []
for i in range(0,len(final_train["POLYLINE"])):
    if final_train["POLYLINE"][i] == '[]':
        k=0
        lists_last_lat.append(k)
    else:
        k = re.sub(r"[[|[|]|]|]]", "", final_train["POLYLINE"][i]).split(",")[-1]
        lists_last_lat.append(k)
        
final_train["lat_last"] = lists_last_lat

print(final_train)

train = final_train.query("lon_last != 0")

train["lon_1st"] = [float(k) for k in train["lon_1st"]]
train["lat_1st"] = [float(k) for k in train["lat_1st"]]
train["lon_last"] = [float(k) for k in train["lon_last"]]
train["lat_last"] = [float(k) for k in train["lat_last"]]
train['call_type_a']= [int(k) for k in train["call_type_a"]]
train['call_type_b'] =[int(k) for k in train["call_type_b"]]
train['call_type_c']= [int(k) for k in train["call_type_c"]]

lists_1st_lon = []
for i in range(0,len(final_test["POLYLINE"])):
    if final_test["POLYLINE"][i] == '[]':
        k=0
        lists_1st_lon.append(k)
    else:
        k = re.sub(r"[[|[|]|]|]]", "", final_test["POLYLINE"][i]).split(",")[0]
        lists_1st_lon.append(k)
        
final_test["lon_1st"] = lists_1st_lon

lists_1st_lat = []
for i in range(0,len(final_test["POLYLINE"])):
    if final_test["POLYLINE"][i] == '[]':
        k=0
        lists_1st_lat.append(k)
    else:
        k = re.sub(r"[[|[|]|]|]]", "", final_test["POLYLINE"][i]).split(",")[1]
        lists_1st_lat.append(k)
        
final_test["lat_1st"] = lists_1st_lat

lists_last_lon = []
for i in range(0,len(final_test["POLYLINE"])):
        if final_test["POLYLINE"][i] == '[]':
            k=0
            lists_last_lon.append(k)
        else:
            k = re.sub(r"[[|[|]|]|]]", "", final_test["POLYLINE"][i]).split(",")[-2]
            lists_last_lon.append(k)

final_test["lon_last"] = lists_last_lon

# last lat
lists_last_lat = []
for i in range(0,len(final_test["POLYLINE"])):
    if final_test["POLYLINE"][i] == '[]':
        k=0
        lists_last_lat.append(k)
    else:
        k = re.sub(r"[[|[|]|]|]]", "", final_test["POLYLINE"][i]).split(",")[-1]
        lists_last_lat.append(k)
        
final_test["lat_last"] = lists_last_lat

test = final_test.query("lon_last != 0")

test["lon_1st"] = [float(k) for k in test["lon_1st"]]
test["lat_1st"] = [float(k) for k in test["lat_1st"]]
test["lon_last"] = [float(k) for k in test["lon_last"]]
test["lat_last"] = [float(k) for k in test["lat_last"]]
test['call_type_a']= [int(k) for k in test["call_type_a"]]
test['call_type_b'] =[int(k) for k in test["call_type_b"]]
test['call_type_c']= [int(k) for k in test["call_type_c"]]

print(test)

test.to_csv("Data\\test_baseline.csv")
train.to_csv("Data\\train_baseline.csv")