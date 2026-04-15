import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_ts_data(data, window_size=6):
    i = 1
    while i < window_size:
        data["co2_{}".format(i)] = data["co2"].shift(-i)
        i += 1
    data["target"] = data["co2"].shift(-i)
    data = data.dropna(axis=0)
    return data

data = pd.read_csv('co2.csv')
data["time"] = pd.to_datetime(data["time"])
data["co2"] = data["co2"].interpolate()

# print(data.info()) #Hơn 50 dòng bị null -> ít dưới 5% có thể drop đi nhưng vì đây là time_series drop đi sẽ gây ngắt quãng -> KHÔNG NÊN DROP
# fig, ax = plt.subplots()
# ax.plot(data["time"],data["co2"])
# ax.set_xlabel("year")
# ax.set_ylabel("co2")
# plt.show()

data = create_ts_data(data)
x = data.drop(["target", "time"], axis=1)  # cột time ở trường hợp này không có nhiều ý nghĩa
y = data["target"]
# Vì đây là dữ liệu Time Series nên phải chia data theo từng giai đoạn để đảm bảo tính liên tục thay vì random như dữ liệu khác
train_ratio = 0.8
num_sample = len(x)
x_train = x[:int(train_ratio * num_sample)]
y_train = y[:int(train_ratio * num_sample)]
x_test = x[int(train_ratio * num_sample):]
y_test = y[int(train_ratio * num_sample):]

reg = LinearRegression() #Không cần phải scaler vì các feature đều từ cột co2 mà ra
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
# print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
# print("R2: {}".format(r2_score(y_test, y_pred)))
'''
Linear Regression
MAE: 0.3605603788359208
MSE: 0.22044947360346367
R2: 0.9907505918201437
Trong thực tế nên dựa và R2 để đánh giá mô hình có tốt hay không vì MAE, MSE vì sai số tuỳ vào biên độ mới đánh giá được mô hình tốt hay không
Ví dụ sai số của giá bán một căn nhà với giá bản của một cân thị lợn
'''
fig, ax = plt.subplots()
ax.plot(data["time"][:int(train_ratio * num_sample)], data["co2"][:int(train_ratio * num_sample)],label="train")
ax.plot(data["time"][int(train_ratio * num_sample):], data["co2"][int(train_ratio * num_sample):],label="test")
ax.plot(data["time"][int(train_ratio * num_sample):], y_pred,label="predict")
ax.set_xlabel("year")
ax.set_ylabel("co2")
ax.legend()
ax.grid()
plt.show()
'''
Lý do Random Forest tệ trong trường hợp này Random Forest bắt nguồn từ Decision Tree và phương pháp dự đoán của mô hình là dựa trên những dữ liệu đã gặp trong quá khứ
Mà dữ liệu lại sử dụng bộ Train là quá khứ, Test là dữ liệu tương lai (tăng dần), Random Forest không thể dự đoán được giá trị chưa gặp bao giờ
Random Forest
MAE: 5.840618421052375
MSE: 52.84716492543475
R2: -1.2173108039385698
'''
#Dự đoán 10 lần tiếp theo
# current_data = [380.5, 390, 390.2, 390.4, 393]
# for i in range(10):
#     print('input data: {}'.format(current_data))
#     prediction = reg.predict([current_data])[0]
#     print('prediction: {}'.format(prediction))
#     current_data = current_data[1:] + [prediction]