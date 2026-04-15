import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
data = pd.read_csv('diabetes.csv')

#####################################
#         Statistic Data            #
#####################################

# stat = data.describe()
# info = data.info()
# countClass = data.groupby('Outcome').size()
# print(data)
# print(stat)
# print(info)
# print(countClass)

# data.hist()
# data.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)

# data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False)

# correlation = data.corr() 
# print(correlation)

# sn.heatmap(data.corr(), annot=True)

# scatter_matrix(data)
# plt.show()

# profile = ProfileReport(data, title='Diabetes Report', explorative=True)
# profile.to_file ('diabetes-report.html')

#####################################
#             Split Data            #
#####################################

target = "Outcome"
x = data.drop(target, axis=1) #1 là cột, 0 là cột (default)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42    ) #0.8 train, 0.2 test
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25) #chia tiếp => 0.6 train, 0.2 val, 0.2 test
# print (x_train.shape) #(614, 8) : 614 sample, 8 cột
# print (y_train.shape) #(614,) : 614 sample, 1 cột - mảng 1 chiều
# print (x_test.shape) #(154, 8) : 154 sample, 8 cột
# print (y_test.shape) #(154,) : 154 sample, 1 cột - mảng 1 chiều
#check tỉ lệ mỗi outcome xem có lệch data quá không
# print(y_train.value_counts())
# print(y_test.value_counts())

#####################################
#        Normalization Data         #
#####################################

scaler = StandardScaler()

# scaler.fit(x_train) #tính u,s theo StandardScaler khai báo ở trên
# x_train = scaler.transform(x_train) #biến đổi dữ liệu sử dụng u,s đã tính đc theo công thức của Scaler

x_train = scaler.fit_transform(x_train) #làm 1 lần cả 2 bước trên, việc fit chỉ dành cho bộ Train để không bị Data Leakage
x_test = scaler.transform(x_test) #transform theo u,s đã fit ở lần gần nhất mà không cần fit sau đó transform từ đầu

#x = scaler.fit_transform(x) #Không bao giờ được fit chung dữ liệu cả bộ Train và Test vì sễ bị Data Leakage

#####################################
#             Run Model             #
#####################################
# model = SVC() #chọn mô hình này vì hệ số tương quan thấp => Nên chọn mô hình phi tuyến
# model = LogisticRegression() #thực tế Logistic lại tốt hơn => thử mô hình thay vì chỉ suy luận trên lý thuyết
model = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=42)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
# for i,j in zip(y_predict, y_test.values):
#     print("Predicted: {}. Actual: {}".format(i, j))
print(classification_report(y_test, y_predict)) #thường dùng weighted avg
"""
weighted avg không phải số liệu duy nhất để nhận định mô hình nào tốt hơn
Ví dụ ở đây cần Recall của class Positive, nên ưu tiên mô hình có Recal của Class 1 hơn
"""