import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances, mean_squared_error
from sklearn.preprocessing import minmax_scale, MultiLabelBinarizer
from sklearn.decomposition import NMF
from random import randint
import kagglehub

# Thiết lập Streamlit
# st.set_page_config(page_title="Hệ thống gợi ý địa điểm theo ngữ cảnh", page_icon=":world_map:", layout="wide")
st.title("Hệ thống gợi ý địa điểm theo ngữ cảnh")
st.markdown("Chọn một User ID để nhận gợi ý địa điểm dựa trên sở thích, thời tiết, mùa và thời gian trong ngày!")

# Tải dữ liệu thời tiết
london_weather_data_path = kagglehub.dataset_download("emmanuelfwerr/london-weather-data")
weather_data = pd.read_csv(os.path.join(london_weather_data_path, 'london_weather.csv'))

# Phân loại thời tiết
def classify_weather(row):
    if row['snow_depth'] > 0 or (row['precipitation'] > 0 and row['mean_temp'] < 0):
        return "Snow"
    elif row['precipitation'] > 0:
        if row['pressure'] < 100000:
            return "Rain (Low Pressure)"
        elif row['min_temp'] < 5:
            return "Cold Rain"
        else:
            return "Rain"
    elif row['sunshine'] > 5 and row['cloud_cover'] < 3:
        if row['max_temp'] > 25:
            return "Hot Sunny"
        elif row['min_temp'] < 5:
            return "Cold Sunny"
        else:
            return "Sunny"
    elif row['cloud_cover'] > 6 and row['sunshine'] < 2:
        if row['min_temp'] < 5:
            return "Cold Cloudy"
        else:
            return "Cloudy"
    else:
        if row['max_temp'] > 25:
            return "Hot Normal"
        elif row['min_temp'] < 5:
            return "Cold Normal"
        else:
            return "Normal"

weather_data['weather_type'] = weather_data.apply(classify_weather, axis=1)
weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y%m%d')

# Mã hóa weather_type
weather_mapping = {
    'Snow': 0, 'Rain (Low Pressure)': 1, 'Cold Rain': 2, 'Rain': 3,
    'Hot Sunny': 4, 'Cold Sunny': 5, 'Sunny': 6, 'Cold Cloudy': 7,
    'Cloudy': 8, 'Hot Normal': 9, 'Cold Normal': 10, 'Normal': 11
}
weather_data['weather_type_encoded'] = weather_data['weather_type'].map(weather_mapping)

# Tải dữ liệu địa điểm
prefiltered_file_path = 'https://drive.usercontent.google.com/u/0/uc?id=140AYsOdzxLUS5hI8l8s1HfccUGU7CKzF&export=download'
data_type = {
    'faves': 'float16',
    'lat': 'float32',
    'lon': 'float32',
    'visit_time': 'datetime64[ns]'
}
LPD = pd.read_csv(prefiltered_file_path, engine='python', sep=',', encoding='utf-8', dtype=data_type, decimal=',')

# Thêm cột month, season, daytime
LPD['month'] = LPD.visit_time.dt.month
def get_season(month):
    if month in [12, 1, 2]:
        return 4
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
LPD['season'] = LPD['month'].apply(get_season)

def get_daytime(hour):
    if 6 <= hour < 18:
        return 1  # day
    elif 18 <= hour < 22:
        return 2  # night
    else:
        return 3  # midnight
LPD['daytime'] = LPD['visit_time'].dt.hour.apply(get_daytime)

# Kết hợp dữ liệu thời tiết
LPD['date'] = pd.to_datetime(LPD['visit_time']).dt.date
weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date
weather_info = weather_data[['date', 'weather_type_encoded']]
LPD = LPD.merge(weather_info, on='date', how='left')
LPD = LPD.set_index(keys=['user_id', 'location_id'])

# Lọc dữ liệu có số lượt ghé thăm > 3
visit_limit = LPD.groupby(level=[0,1])['visit_time'].count()
visit_limit = visit_limit[visit_limit > 3]
mask = LPD.index.isin(visit_limit.index)
X = LPD[mask]
y = X.index.get_level_values(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=70)

# Tạo ma trận đánh giá
train_rating = X_train.groupby(['location_id', 'user_id'])['visit_time'].count().reset_index(name='rating')
def normalize(df):
    df['rating'] = minmax_scale(df.rating, feature_range=(1,5))
    return df
train_rating = normalize(train_rating)
r_df = train_rating.pivot_table(index='user_id', columns='location_id', values='rating', fill_value=0)
r_df_display = train_rating.pivot_table(index='user_id', columns='location_id', values='rating', fill_value="?")

# Tính tỷ lệ rỗng
def calSparcity(m):
    m_filled = m.replace("?", 0).astype(float) if isinstance(m, pd.DataFrame) and m.values.dtype == object else m.fillna(0)
    non_zeros = np.count_nonzero(m_filled) / np.prod(m_filled.shape) * 100
    sparcity = 100 - non_zeros
    return round(sparcity, 2)

sparcity = calSparcity(r_df)
st.subheader(f"Ma trận đánh giá (hiển thị '?' cho địa điểm chưa ghé thăm, tỷ lệ rỗng: {sparcity}%)")
st.dataframe(r_df_display.head(10))

# Ma trận tương đồng người dùng cải tiến
def improved_asym_cosine(m, mf=False, **kwarg):
    cosine = cosine_similarity(m)
    def asymCo(X, Y):
        co_rated_item = np.intersect1d(np.nonzero(X), np.nonzero(Y)).size
        coeff = co_rated_item / np.count_nonzero(X)
        return coeff
    asym_ind = pairwise_distances(m, metric=asymCo)
    sorensen = 1 - pairwise_distances(np.array(m, dtype=bool), metric='dice')
    def usrInfCo(m):
        binary = m.transform(lambda x: x >= x[x != 0].mean(), axis=1) * 1
        res = pairwise_distances(binary, metric=lambda x, y: (x * y).sum() / y.sum() if y.sum() != 0 else 0)
        return res
    usr_inf_ind = usrInfCo(m)
    similarity_matrix = np.multiply(np.multiply(cosine, asym_ind), np.multiply(sorensen, usr_inf_ind))
    usim = pd.DataFrame(similarity_matrix, m.index, m.index)
    if mf:
        binary = np.invert(usim.values.astype(bool)) * 1
        model = NMF(**kwarg)
        W = model.fit_transform(usim)
        H = model.components_
        factorized_usim = np.dot(W, H) * binary + usim
        usim = pd.DataFrame(factorized_usim, m.index, m.index)
    return usim

s_df = improved_asym_cosine(r_df)

# Tạo ma trận ngữ cảnh
listcontext = ['daytime', 'season', 'weather_type_encoded']
X_train['context'] = X_train.apply(lambda x: tuple(x[col] for col in listcontext), axis=1)
contexts = X_train.reset_index()[['location_id', 'user_id', 'context']]
IF = contexts.groupby(['location_id', 'context'])['context'].count() / contexts.groupby(['context'])['context'].count()
IDF = np.log10(contexts.groupby(['location_id', 'user_id'])['user_id'].count().sum() / contexts.groupby(['location_id'])['user_id'].count())
contexts_weight = (IF * IDF).to_frame().rename(columns={0: 'weight'})
lc_df = contexts_weight.pivot_table(index='context', columns='location_id', values='weight', fill_value=0)
cs_df = pd.DataFrame(cosine_similarity(lc_df), index=lc_df.index, columns=lc_df.index)
context_sparcity = calSparcity(cs_df)

# Hàm Collaborative Filtering
def CF(user_id, location_id, s_matrix):
    r = np.array(r_df)
    s = np.array(s_matrix)
    users = r_df.index
    locations = r_df.columns
    l = np.where(locations == location_id)[0]
    u_idx = np.where(users == user_id)[0]
    means = np.array([np.mean(row[row != 0]) for row in r])
    if location_id in r_df:
        idx = np.nonzero(r[:, l])[0]
        sim_scores = s[u_idx, idx].flatten()
        if idx.any():
            sim_ratings = r[idx, l]
            sim_means = means[idx]
            numerator = (sim_scores * (sim_ratings - sim_means)).sum()
            denominator = np.absolute(sim_scores).sum()
            weight = (numerator / denominator) if denominator != 0 else 0
            wmean_rating = means[u_idx] + weight
            wmean_rating = wmean_rating[0]
        else:
            wmean_rating = 0
    else:
        wmean_rating = 0
    return wmean_rating

# Hàm CaCF với hậu xử lý
def CaCF_Post(user_id, location_id, s_matrix, c_current, delta):
    initial_pred = CF(user_id, location_id, s_matrix)
    if location_id in r_df:
        r = np.array(r_df)
        users = r_df.index
        locations = r_df.columns
        l = np.where(locations == location_id)[0]
        c_profile = contexts
        all_cnx = contexts.context.unique().tolist()
        c = np.array(c_profile)
        u_idx = np.where(users == user_id)[0]
        c_current = tuple(c_current)
        l_cnx = np.array(c_profile.loc[c_profile.location_id == location_id, ['user_id', 'context']])
        if c_current in all_cnx:
            cnx_scores = np.array([[uid, cs_df[c_current][cx]] for uid, cx in l_cnx])
            filtered_scores = cnx_scores[cnx_scores[:, 1].astype(float) > delta]
            visit_prob = len(filtered_scores) / len(cnx_scores) if len(cnx_scores) > 0 else 1
        else:
            visit_prob = 1
        return initial_pred * visit_prob
    return initial_pred

# Tạo dữ liệu kiểm tra
test_rating = X_test.groupby(['location_id', 'user_id'])['visit_time'].count().reset_index(name='rating')
test_rating = normalize(test_rating)
r_df_test = test_rating.pivot_table(index='user_id', columns='location_id', values='rating', fill_value=0)

# Hàm dự đoán
def EACOS_CaCF_Post(user_id, location_id, c_current, delta):
    return CaCF_Post(user_id, location_id, s_df, c_current, delta)

def predict(target_user, model, option=None):
    true = r_df_test.loc[target_user]
    if option:
        pred_val = []
        for l in true.index:
            delta = option.get('delta')
            c_current = tuple(X_test.xs(target_user)[listcontext].head(1).values[0])
            r = model(user_id=target_user, location_id=l, c_current=c_current, delta=delta)
            pred_val.append(r)
    else:
        pred_val = [model(user_id=target_user, location_id=l) for l in true.index]
    pred = pd.Series(pred_val, index=true.index)
    return pred

# Giao diện gợi ý
st.subheader("Gợi ý địa điểm")
user_id = st.selectbox("Chọn User ID", r_df_test.index)
delta = st.slider("Ngưỡng tương đồng ngữ cảnh (δ)", min_value=0.05, max_value=0.95, step=0.05, value=0.3)
# Đánh giá hiệu suất
def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

def mean_average_precision(true, pred, k=10):
    relevant = 1
    true_lists = [r[1][r[1] > relevant].dropna().index.tolist() for r in true.iterrows()]
    pred_lists = [r[1][r[1] > relevant].dropna().sort_values(ascending=False).index.tolist() for r in pred.iterrows()]
    def apk(actual, predicted, k=10):
        if not actual:
            return 0.0
        predicted = predicted[:k]
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(actual), k)
    scores = [apk(a, p, k) for a, p in zip(true_lists, pred_lists)]
    return sum(scores) / len(scores) if scores else 0.0

def predict_all(model, option=None):
    users = r_df_test.index
    locations = r_df_test.columns
    pred = np.zeros(r_df_test.shape)
    for i in range(len(users)):
        uid = users[i]
        for j in range(len(locations)):
            lid = locations[j]
            if option:
                delta = option.get('delta')
                c_current = tuple(X_test.xs(uid)[listcontext].head(1).values[0])
                pred[i,j] = model(user_id=uid, location_id=lid, c_current=c_current, delta=delta)
            else:
                pred[i,j] = model(user_id=uid, location_id=lid)
    return pd.DataFrame(pred, index=users, columns=locations)


if st.button("Gợi ý địa điểm"):
    options = {'delta': delta}
    pred = predict(user_id, EACOS_CaCF_Post, option=options)
    true = r_df_test.loc[user_id]
    prediction = pd.DataFrame({'true': true, 'pred': pred})
    top_10 = prediction.nlargest(10, 'pred')
    st.subheader(f"Top 10 địa điểm gợi ý cho User ID {user_id}")
    def item_relevancy(col):
        relevant = 1
        r_color = 'background-color: lime'
        nr_color = 'background-color: red'
        res = []
        for v in col:
            if v > relevant:
                res.append(r_color)
            elif (v > 0) and (v <= relevant):
                res.append(nr_color)
            else:
                res.append('')
        return res
    st.dataframe(top_10.style.apply(item_relevancy))


    # Đánh giá ảnh hưởng của delta
    st.subheader("Ảnh hưởng của ngưỡng tương đồng ngữ cảnh (δ)")
    deltas = np.arange(0.05, 1, 0.05)
    eval_scores = []
    options = {
        'delta': .3
    }
    for d in deltas:
        options['delta'] = d
        pred = predict_all(EACOS_CaCF_Post, option=options)
        precision = mean_average_precision(r_df_test, pred)
        eval_scores.append(precision)

    d_eval = pd.DataFrame(eval_scores, index=deltas, columns=['precision'])
    d_precision = go.Figure(data=[
        go.Scatter(
            name='MAP',
            x=d_eval.index,
            y=d_eval['precision'],
            text=d_eval['precision'],
            line_shape='spline',
            mode='lines+markers',
            marker=dict(size=8)
        )
    ])
    d_precision.update_layout(
        title='Ảnh hưởng của ngưỡng tương đồng ngữ cảnh đến chất lượng gợi ý',
        xaxis=dict(title='Ngưỡng tương đồng ngữ cảnh (δ)'),
        yaxis=dict(title='MAP@10'),
        template='plotly_white'
    )
    st.plotly_chart(d_precision)