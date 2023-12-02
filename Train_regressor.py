import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("Data\\data_5.csv")
test = pd.read_csv("Data\\test_our.csv")

encoder = OneHotEncoder(handle_unknown='ignore')
encoder_df = pd.DataFrame(encoder.fit_transform(train[['CALL_TYPE']]).toarray())
train = train.join(encoder_df)
train.rename(columns={0:'call_type_a', 1:'call_type_b',2:'call_type_c'}, inplace=True)

train['call_type_a']= [int(k) for k in train["call_type_a"]]
train['call_type_b'] =[int(k) for k in train["call_type_b"]]
train['call_type_c']= [int(k) for k in train["call_type_c"]]

del train['CALL_TYPE']
del test['CALL_TYPE']

train['ORIGIN_CALL'] = train[['ORIGIN_CALL']].fillna('')
train['ORIGIN_STAND'] = train[['ORIGIN_STAND']].fillna('')
test['ORIGIN_CALL'] = test[['ORIGIN_CALL']].fillna('')
test['ORIGIN_STAND'] = test[['ORIGIN_STAND']].fillna('')


ml_train = train.copy()

# Origin_call
def origin_call_flg(x):
    if x["ORIGIN_CALL"] == None:
        res = 0
    else:
        res = 1
    return res
ml_train["ORIGIN_CALL"] = ml_train.apply(origin_call_flg, axis=1)

# Origin_stand
def origin_stand_flg(x):
    if x["ORIGIN_STAND"] == None:
        res = 0
    else:
        res=1
    return res
ml_train["ORIGIN_STAND"] = ml_train.apply(origin_stand_flg, axis=1)


# Missing data
def miss_flg(x):
    if x["MISSING_DATA"] == "False":
        res = 0
    else:
        res = 1
    return res
ml_train["MISSING_DATA"] = ml_train.apply(miss_flg, axis=1)

ml_test = test.copy()

# Origin_call
def origin_call_flg(x):
    if x["ORIGIN_CALL"] == None:
        res = 0
    else:
        res = 1
    return res
ml_test["ORIGIN_CALL"] = ml_test.apply(origin_call_flg, axis=1)

# Origin_stand
def origin_stand_flg(x):
    if x["ORIGIN_STAND"] == None:
        res = 0
    else:
        res=1
    return res
ml_test["ORIGIN_STAND"] = ml_test.apply(origin_stand_flg, axis=1)


# Missing data
def miss_flg(x):
    if x["MISSING_DATA"] == "False":
        res = 0
    else:
        res = 1
    return res
ml_test["MISSING_DATA"] = ml_test.apply(miss_flg, axis=1)


ml_train = ml_train.sample(136000)

print(ml_train.info())
X = ml_train[["call_type_a","call_type_b","call_type_c",'ORIGIN_CALL','ORIGIN_STAND', 'MISSING_DATA', 'x_from', 'y_from',
               'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4', 'x_5', 'y_5','Year','Month','Week','Day','Hour','Min','Weekday']]
Y = ml_train[["x_to","y_to"]]

#Y_Test = ml_test[["lon_last","lat_last"]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

lr = MultiOutputRegressor(LinearRegression(n_jobs=1))

lr = lr.fit(X_train, Y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print("Mean Square Error on training Data:{}".format(mean_squared_error(Y_train, y_train_pred)))
print("Mean Square Error on testing Data:{}".format(mean_squared_error(Y_test, y_test_pred)))
print("R2 score train:{}".format(r2_score(Y_train, y_train_pred)))
print("R2 score test:{}".format(r2_score(Y_test, y_test_pred)))
X_Test = ml_test[["call_type_a","call_type_b","call_type_c",'ORIGIN_CALL','ORIGIN_STAND', 'MISSING_DATA', 'x_from', 'y_from',
               'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'x_4', 'y_4', 'x_5', 'y_5','Year','Month','Week','Day','Hour','Min','Weekday']]
y_Test_pred = lr.predict(X_Test)

output = pd.DataFrame({"TRIP_ID":test["TRIP_ID"], "LATITUDE":y_Test_pred.T[1],"LONGITUDE":y_Test_pred.T[0]})
output.to_csv("Data\\submission_ourregressor_adddate.csv")