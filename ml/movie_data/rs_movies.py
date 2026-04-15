import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('movies.csv', encoding='latin-1', sep='\t', usecols=["title", "genres"])
data["genres"] = data["genres"].apply(lambda title: title.replace("|", " ").replace("-", "")) #replace("-", " ") vì trong vocab có từ sci-fi và bị chia thành 2 từ sci và fi
#bộ này không chia dữ liệu train test vì không có label để đánh giá
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["genres"])
tfidf_matrix_df =  pd.DataFrame(tfidf_matrix.todense(), index=data.index, columns=vectorizer.get_feature_names_out())
print(vectorizer.vocabulary_)
print(len(vectorizer.vocabulary_))
print(tfidf_matrix.shape)

cosine_similarity = cosine_similarity(tfidf_matrix) #càng gần 1 thì cos càng gần 0 => 2 vector càng tương đồng
cosine_similarity_df = pd.DataFrame(cosine_similarity, index=data["title"], columns=data["title"])

input_movie = "Toy Story (1995)"
top_k = 20
relevant_row = cosine_similarity_df.loc[input_movie, :]
results = relevant_row.sort_values(ascending=False)[:top_k].to_frame(name="score").reset_index()