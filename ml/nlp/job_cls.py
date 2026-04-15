import pandas as pd
from scipy.special.cython_special import eval_chebyc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.model_selection import GridSearchCV

def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location

data = pd.read_excel('job.ods', engine='odf', dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)
# print(len(data["function"].unique())) #có 19 unique không nhất thiết phải dùng Tfidf có thể dùng OneHot
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) #test_size chỉ dám bảo tỉ lệ với tổng sample, phải thêm stratify để đảm bảo cho mỗi feature nữa

# print(y_train.value_counts())
#####################################
#            Balance Data           #
#####################################
# ros = RandomOverSampler(random_state=0, sampling_strategy={
ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
    "director_business_unit_leader": 500,
    "specialist": 500,
    "managing_director_small_medium_company": 500
}) #mặc định nếu không truyền strategy thì sẽ sampling theo feature có nhiều data nhất
x_train,y_train = ros.fit_resample(x_train, y_train) #Chỉ OverSampling với bộ train, để không bị data leakage
# print(y_train.value_counts())

#####################################
#     Test Preprocessing Các FT     #
#####################################
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)) #với title ngắn nên có thể để default ngram_rang(1,1) nhưng với description dài để ngram_range(1,2) để lưu thông tin liên quan đến thứ tự nữa tốt hơn
# result = vectorizer.fit_transform(x_train['title'])
# result = vectorizer.fit_transform(x_train['description'])
# result = pd.DataFrame(result.todense()) #chỉ để show data
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(result.shape)

#####################################
#       Preprocessing Feature       #
#####################################
preprocessor = ColumnTransformer(transformers=[
    ("title_fr", TfidfVectorizer(stop_words="english"), "title"),
    ("location_ft", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description_ft", TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=0.01, max_df=0.95), "description"),
    ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_ft", TfidfVectorizer(stop_words="english"), "industry"),
])
reg = Pipeline(steps=[
    ('preprocessing', preprocessor),
    # ('feature_selector', SelectKBest(chi2, k=800)),
    ('feature_selector',SelectPercentile(chi2, percentile=5)),
    ('model', RandomForestClassifier())
])
parameters = {
    'model__n_estimators': [100,200,300],
    'model__criterion':['gini','entropy','log_loss'],
    'feature_selector__percentile': [1,5,10],
}

# result = reg.fit_transform(x_train, y_train)
# print(result.shape)
# reg.fit(x_train, y_train)
# y_pred = reg.predict(x_test)

grid_search = GridSearchCV(estimator=reg, param_grid=parameters, cv=4, verbose=2, scoring='recall_weighted', n_jobs=6)
grid_search.fit(x_train, y_train)
y_pred = grid_search.predict(x_test)
print(classification_report(y_test, y_pred))

#Trước khi thêm min_df và max_df 850k features
"""
   accuracy                               0.69      1615
   macro avg          0.50      0.30      0.32      1615
   weighted avg       0.68      0.69      0.64      1615
"""
#Sau khi thêm min_df và max_df, có SelectKBest 8000 features
"""
    accuracy                               0.74      1615
    macro avg          0.68      0.34      0.38      1615
    weighted avg       0.73      0.74      0.70      1615
"""
#có SelectKbest 800 features
"""
    accuracy                               0.75      1615
    macro avg          0.53      0.34      0.37      1615
    weighted avg       0.75      0.75      0.72      1615 
"""
#Sử dụng SelectPercentile với 5% = 400 features
"""
    accuracy           0.76      1615
    macro avg          0.69      0.38      0.43      1615
    weighted avg       0.76      0.76      0.74      1615
"""
#Có một cách để tăng hiệu quả cho case này thì có thể gộp các class level có số lượng ít với nhau nhưng phải đảm bảo gộp các class LIÊN TIẾP nhau