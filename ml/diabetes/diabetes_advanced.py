import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv('diabetes.csv')
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
scaler.fit_transform(x_train)
scaler.transform(x_test)

#k-fold cross validation
# parameters = {
#     'n_estimators': [100,200,300],
#     'criterion':['gini','entropy','log_loss'],
#     #Không được cho random_state ví dụ như 'random_state':[42,42,43] vào đây vì nó phải cố định để đánh giá công bằng
# }
# grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=parameters, cv=4, verbose=2, scoring='recall')
# grid_search.fit(x_train, y_train)
# print(grid_search.best_estimator_)
# print(grid_search.best_params_)
# print(grid_search.best_score_) #default Accuracy
# y_predict = grid_search.predict(x_test)
# print(classification_report(y_test, y_predict))

#####################################
#           Lazy Predict            #
#####################################
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)