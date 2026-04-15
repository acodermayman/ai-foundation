import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_ts_data(data, window_size=5, target_size=3):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    i = 1
    while i <= target_size:
        data["target_{}".format(i)] = data["target"].shift(-i-window_size)
        i += 1
    data = data.dropna(axis=0)
    return data

data = pd.read_csv('co2.csv')
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()
window_size = 5
target_size = 3
data = create_ts_data(data,window_size,target_size)
targets = ["target_{}".format(i+1) for i in range(target_size)]
x = data.drop(["time"] + targets, axis=1)
y = data[targets]

train_ratio = 0.8
num_sample = len(x)
x_train = x[:int(train_ratio * num_sample)]
y_train = y[:int(train_ratio * num_sample)]
x_test = x[int(train_ratio * num_sample):]
y_test = y[int(train_ratio * num_sample):]

regs = [LinearRegression() for _ in range(target_size)]
for i,reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i+1)])
r2 = []
mse = []
mae = []
y_preds = []
for i,reg in enumerate(regs):
    y_pred = reg.predict(x_test)
    y_preds.append(y_pred)
    r2.append(r2_score(y_test["target_{}".format(i+1)], y_pred))
    mse.append(mean_squared_error(y_test["target_{}".format(i+1)], y_pred))
    mae.append(mean_absolute_error(y_test["target_{}".format(i+1)], y_pred))
print("MAE: {}".format(mae))
print("MSE: {}".format(mse))
print("R2: {}".format(r2))

#chưa biết cách fix
fig, ax = plt.subplots()
ax.plot(data["time"][:int(train_ratio * num_sample)], data["co2"][:int(train_ratio * num_sample)],label="train")
ax.plot(data["time"][int(train_ratio * num_sample):], data["co2"][int(train_ratio * num_sample):],label="test")
for i,y_pred in enumerate(y_preds):
    ax.plot(data["time"][int(train_ratio * num_sample):], y_pred,label="predict_{}".format(i+1))
ax.set_xlabel("year")
ax.set_ylabel("co2")
ax.legend()
ax.grid()
plt.show()

