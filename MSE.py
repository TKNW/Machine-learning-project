import pandas as pd
from sklearn.metrics import mean_squared_error

submission = pd.read_csv("Data\\submission.csv")
Origin = pd.read_csv("Data\\test_our.csv")
input1 = pd.DataFrame({'x_to':submission["LONGITUDE"],'y_to':submission['LATITUDE']})
input2 = pd.DataFrame({'x_to':Origin['x_to'],'y_to':Origin['y_to']})
print("MSE = {}".format(mean_squared_error(input1,input2)))
input1 = pd.DataFrame({'x_to':submission["LATITUDE"],'y_to':submission['LONGITUDE']})
print("MSE = {}".format(mean_squared_error(input1,input2)))