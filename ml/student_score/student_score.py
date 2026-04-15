"""
chọn bất kì cột math, reading, writing làm target
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score #Metric Coefficient Of Determination phổ biến nhất cho bài toán Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyRegressor

#####################################
#      Read And Analysis File       #
#####################################
data = pd.read_csv('student_score.xls')
# profile = ProfileReport(data, title="Score Report", explorative=True)

# print(data[["math score", "reading score", "writing score"]].corr())
target = "writing score"

x = data.drop(target, axis=1)
y = data[target]

#####################################
#         Preprocessing Data        #
#####################################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""
 Trên lý thuyết thì loại bỏ outliers rất quan trọng nhưng trong thực tế thì không phải lúc nào
 loại bỏ outliers cũng tốt trừ khi dữ liệu đã quá rõ ràng, và cũng có thể bỏ qua bước này vì 
 bước chọn lọc dữ liệu sau này cũng phần nào bao gồm cả outliers
"""

# print(data["gender"].unique()) #không phải lúc nào gender cũng là boolean

"""
race/ethnicity là nominal feature & Group A B C ý nghĩa là tránh PHÂN BIỆT CHỦNG TỘC (domain knowledge)
"""
# print(data["parental level of education"].unique())

"""
#thay vì transform từng bước ta có thể làm pineline như dưới
imputer = SimpleImputer(strategy="median", mising_values=-1)
x_train[["math score", "reading score"]] = imputer.fit_transform(x_train[["math score", "reading score"]])
scaler = StandardScaler()
x_train[["math score", "reading score"]] = scaler.fit_transform(x_train[["math score", "reading score"]])
"""
numeric_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=-1, strategy="median")),
    ('scaler', StandardScaler())
])
# result=numeric_transform.fit_transform(x_train[["math score", "reading score"]])
# for i,j in zip(x_train[["math score", "reading score"]].values, result):
#     print("Before {}; After {}".format(i, j))
"""
 cột bachelor's degree xử lý bằng cách định nghĩa theo Ordinal Feature gán theo số thứ tự 0->5 sử dụng Ordinal Encoder  
"""
categories_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                     "master's degree"]
gender_values = ['male', 'female']
lunch_values = x_train["lunch"].unique()
test_values = x_test["test preparation course"].unique()
ordinal_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OrdinalEncoder(categories=[categories_values,gender_values,lunch_values,test_values]))  # nếu không defind categories thì sẽ lấy theo AlphaB
])
"""
Các cột có 2 giá trị (boolean) như "gender", "lunch", "test preparation course" nên OrdinalEncoder để tiết kiệm bộ nhớ (chỉ tạo ra 1 cột) 
Nếu dùng OneHotEncoder cũng được nhưng sẽ tạo ra 2 cột tốn bộ nhớ hơn
"""
# result=ordinal_transform.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
# for i,j in zip(x_train[["parental level of education", "gender", "lunch", "test preparation course"]].values, result):
#     print("Before {}; After {}".format(i, j))
nominal_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder()) #có thể thêm param sparse_output=false
])
# result=nominal_transform.fit_transform(x_train[["race/ethnicity"]])
# for i,j in zip(x_train[["race/ethnicity"]].values, result):
#     print("Before {}; After {}".format(i, j)) #j → kết quả mã hóa one-hot
preprocessing = ColumnTransformer([
    ("numeric_feature", numeric_transform,["math score", "reading score"]),
    ("ordinal_feature", ordinal_transform,["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_feature", nominal_transform,["race/ethnicity"]),
])

reg = Pipeline(steps=[
    ('preprocessing', preprocessing),
    # ('model', LinearRegression())
    ('model', RandomForestRegressor())
])
# reg.fit(x_train, y_train)
# y_pred = reg.predict(x_test)
# for i,j in zip(y_test, y_pred):
#     print("Predicted Value {} Actual Value {}".format(j,i))
# print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
# print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
# print("R2: {}".format(r2_score(y_test, y_pred)))
"""
    Linear Regression
MAE: 3.2039447691582152
MSE: 14.980822041816777
R2: 0.9378432907399291
    RandomForestRegressor
MAE: 3.6351150000000003
MSE: 20.367671040277777
R2: 0.9154927944794022
** Việc RandomForestRegressor phức tạp hơn nhưng kết quả kém hơn LinearRegression bởi vì 
có feature là Reading, Math có hệ số tương quan lớn với Target là Writing
"""
parameters = {
    'preprocessing__numeric_feature__imputer__strategy': ['median', 'mean'],
    'model__n_estimators': [100,200,300],
    'model__criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'model__max_depth': [None,2,5]
}
grid_search = GridSearchCV(estimator=reg, param_grid=parameters, cv=4, verbose=2, scoring='r2', n_jobs=6)
# grid_search = GridSearchCV(estimator=reg, param_grid=parameters, cv=4, verbose=2, scoring='neg_mean_absolute_error')
"""
    một số metrics có prefix lại l "neg" như neg_mean_absolute_error bởi vì Mean Absolute Error càng nhỏ càng tốt
    còn R2 càng lớn càng tốt, để đưa về chung hệ quy chiếu là càng lớn càng tốt nên thêm Negative (-) vào trước
"""
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred = grid_search.predict(x_test)
print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
print("R2: {}".format(r2_score(y_test, y_pred)))
"""
 Khi Có quá nhiều bộ param để thử có thể sử dụng RandomSearch để tìm bộ tốt nhất trong các bộ random để đỡ tốn thời gain
"""
#random_search = RandomizedSearchCV(estimator=reg, param_distributions=parameters, n_iter=20, cv=4, verbose=2, scoring='r2', n_jobs=6)

# rgs = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = rgs.fit(x_train, x_test, y_train, y_test)
