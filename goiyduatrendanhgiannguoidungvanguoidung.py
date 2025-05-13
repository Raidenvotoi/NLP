import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# st.set_page_config(page_title="Hệ thống gợi ý phim dựa trên người dùng và người dùng", page_icon=":thumbsup:", layout="wide")

# Thiết lập Streamlit
st.title("Hệ thống gợi ý phim dựa trên người dùng")
st.markdown("Chọn một User ID để xem ma trận đánh giá và nhận gợi ý phim dựa trên sở thích của những người dùng tương tự!")

# Tải dữ liệu
import kagglehub

# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
# Tải dữ liệu
movies = pd.read_csv(path+'/movies_metadata.csv')
ratings = pd.read_csv(path+'/ratings_small.csv')
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')

# Hàm lấy tiêu đề phim
def get_movie_title(movie_id):
    try:
        return movies.loc[movies['id'] == movie_id, 'title'].values[0]
    except (IndexError, TypeError):
        return None

# Chuẩn bị ma trận đánh giá
ratings_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Ma trận để tính toán (điền 0)
ratings_filled = ratings_matrix.fillna(0)

# Ma trận để hiển thị (điền "?")
ratings_display = ratings_matrix.fillna("?")

# Tính độ tương đồng người dùng
user_similarity = cosine_similarity(ratings_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# Hàm gợi ý phim
def score_unknown_movies(user_id, user_similarity_df, ratings_matrix, movie_rating_counts, m=50, top_n=5):
    if user_id not in ratings_matrix.index:
        st.error(f"User ID {user_id} không tồn tại trong cơ sở dữ liệu.")
        return []
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:20]
    unknown_movies = ratings_matrix.columns[ratings_matrix.loc[user_id].isnull()]
    
    sim_scores = user_similarity_df[user_id][similar_users]
    ratings_subset = ratings_matrix.loc[similar_users, unknown_movies]
    weighted_scores = ratings_subset.mul(sim_scores, axis=0).sum(axis=0)
    sim_sums = ratings_subset.notnull().mul(sim_scores, axis=0).sum(axis=0)
    
    scores = weighted_scores / (sim_sums + 1e-8)
    confidence = movie_rating_counts[unknown_movies] / (movie_rating_counts[unknown_movies] + m)
    final_scores = scores * confidence
    
    return pd.DataFrame({
        'movieId': unknown_movies,
        'score': final_scores
    }).sort_values('score', ascending=False).head(top_n)

# Giao diện Streamlit
st.subheader("Ma trận đánh giá (hiển thị '?' cho phim chưa đánh giá)")
st.dataframe(ratings_display.head(10))  # Hiển thị 10 người dùng đầu tiên

user_id = st.selectbox("Chọn User ID", ratings_matrix.index)
if st.button("Gợi ý phim"):
    movie_rating_counts = ratings_matrix.count()
    result = score_unknown_movies(user_id, user_similarity_df, ratings_matrix, movie_rating_counts, m=50)
    if not result.empty:
        st.subheader(f"Gợi ý phim cho User ID {user_id}")
        recommendations_df = result.copy()
        recommendations_df['title'] = recommendations_df['movieId'].apply(get_movie_title)
        recommendations_df = recommendations_df[['title', 'score']].dropna()
        st.dataframe(recommendations_df.reset_index(drop=True))
    else:
        st.warning("Không tìm thấy phim phù hợp để gợi ý.")