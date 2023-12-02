import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

train = pd.read_csv("Data\\train_baseline.csv")
test = pd.read_csv("Data\\test_baseline.csv")

del train['CALL_TYPE']
del test['CALL_TYPE']

train['ORIGIN_CALL'] = train[['ORIGIN_CALL']].fillna('')
train['ORIGIN_STAND'] = train[['ORIGIN_STAND']].fillna('')
test['ORIGIN_CALL'] = test[['ORIGIN_CALL']].fillna('')
test['ORIGIN_STAND'] = test[['ORIGIN_STAND']].fillna('')


train["delta_lon"] = train["lon_last"] - train["lon_1st"]
train["delta_lat"] = train["lat_last"] - train["lat_1st"]

test["delta_lon"] = test["lon_last"] - test["lon_1st"]
test["delta_lat"] = test["lat_last"] - test["lat_1st"]

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

print(ml_train)
print(ml_test)

ml_train = ml_train.sample(136000)
X = ml_train[["call_type_a","call_type_b","call_type_c",'ORIGIN_CALL','ORIGIN_STAND', 'MISSING_DATA', 'lon_1st', 'lat_1st', 'delta_lon', 'delta_lat']]
Y = ml_train[["lon_last","lat_last"]]

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
X_Test = ml_test[["call_type_a","call_type_b","call_type_c",'ORIGIN_CALL','ORIGIN_STAND', 'MISSING_DATA', 'lon_1st', 'lat_1st', 'delta_lon', 'delta_lat']]
y_Test_pred = lr.predict(X_Test)

output = pd.DataFrame({"TRIP_ID":test["TRIP_ID"], "LATITUDE":y_Test_pred.T[1],"LONGITUDE":y_Test_pred.T[0]})
print(output)
output.to_csv("Data\\submission.csv")
