import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Thiết lập Streamlit
st.set_page_config(page_title="Hệ thống gợi ý phim dựa trên Item-Based Collaborative Filtering", page_icon=":movie_camera:", layout="wide")
st.title("Hệ thống gợi ý phim dựa trên Item-Based Collaborative Filtering")
st.markdown("Chọn một User ID để xem ma trận đánh giá và nhận gợi ý phim dựa trên độ tương đồng giữa các phim!")
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

# Tính độ tương đồng giữa các phim
item_similarity = pd.DataFrame(
    cosine_similarity(ratings_filled.T),
    index=ratings_filled.columns,
    columns=ratings_filled.columns
)

# Hàm dự đoán và gợi ý phim
def predict_item_based(user_id, ratings_matrix, item_similarity, top_n=5):
    if user_id not in ratings_matrix.index:
        st.error(f"User ID {user_id} không tồn tại trong cơ sở dữ liệu.")
        return []
    # Nếu user_id không có trong ratings_matrix, trả về danh sách rỗng
    user_ratings = ratings_matrix.loc[user_id] # Lấy đánh giá của user
    unseen_movies = user_ratings[user_ratings.isnull()].index # Lấy danh sách phim chưa xem
    predicted_ratings = []

    for movie in unseen_movies:
        rated_movies = user_ratings[user_ratings.notnull()].index # Lấy danh sách phim đã xem
        sim_scores = item_similarity.loc[movie, rated_movies] # Lấy độ tương đồng giữa phim chưa xem và các phim đã xem
        user_scores = user_ratings[rated_movies] # Lấy điểm số của user cho các phim đã xem
        
        if sim_scores.sum() == 0:
            predicted_score = 0
        else:
            predicted_score = np.dot(sim_scores, user_scores) / sim_scores.sum() # Tính điểm dự đoán bangwf công thức weighted average
        
        predicted_ratings.append((movie, predicted_score))
    
    return sorted(predicted_ratings, key=lambda x: x[1], reverse=True)[:top_n]

# Giao diện Streamlit
st.subheader("Ma trận đánh giá (hiển thị '?' cho phim chưa đánh giá)")
st.dataframe(ratings_display.head(10))  # Hiển thị 10 người dùng đầu tiên

user_id = st.selectbox("Chọn User ID", ratings_matrix.index)
if st.button("Gợi ý phim"):
    recommendations = predict_item_based(user_id=user_id, ratings_matrix=ratings_matrix, item_similarity=item_similarity)
    if recommendations:
        st.subheader(f"Gợi ý phim cho User ID {user_id}")
        recommendations_df = pd.DataFrame(recommendations, columns=['movieId', 'score'])
        recommendations_df['title'] = recommendations_df['movieId'].apply(get_movie_title)
        recommendations_df = recommendations_df[['title', 'score']].dropna()
        st.dataframe(recommendations_df.reset_index(drop=True))
    else:
        st.warning("Không tìm thấy phim phù hợp để gợi ý.")