import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import kagglehub

# Download latest version
path = kagglehub.dataset_download("huhuyngun/vietnamese-chatbot-ver2")

print("Path to dataset files:", path)

qa_dataset=pd.read_csv(path+"/vi-QA.csv")
with open('data\question_embeddings.pkl', 'rb') as f:
       question_embeddings = pickle.load(f)
user_question = "Có crush ai không	"
user_embedding = model.encode(user_question)
import numpy as np
# scores = util.cos_sim(user_embedding, question_embeddings)
# best_idx = np.argmax(scores.cpu().numpy())
# print("Trả lời:", qa_dataset['answers'][best_idx])





import streamlit as st

import requests
st.title("Hệ thống chatbot dựa trên SentenceTransformer")
def get_answer(user):
    user_embedding = model.encode(user)
    scores = util.cos_sim(user_embedding, question_embeddings)
    best_idx = np.argmax(scores.cpu().numpy())
    return qa_dataset['answers'][best_idx]


messages = st.container(height=400)

if "store_messages_user" not in st.session_state:
    messages.chat_message("assistant").write("Chào bạn, tôi là chatbot hỗ trợ bạn trong việc tìm kiếm thông tin. Bạn có thể hỏi tôi bất cứ điều gì liên quan đến dữ liệu của chúng tôi.")
    st.session_state.store_messages_user = []
    st.session_state.store_messages_bot = ["Chào bạn, tôi là chatbot hỗ trợ bạn trong việc tìm kiếm thông tin. Bạn có thể hỏi tôi bất cứ điều gì liên quan đến dữ liệu của chúng tôi."]
    
else :
    store_messages_user = st.session_state.get("store_messages_user", [])
    store_messages_bot = st.session_state.get("store_messages_bot", [])




if "store_messages_user" not in st.session_state:
    
    print("dsfsdf")
    for i in range(len(store_messages_user)):
        st.write("dsfsdf")
        messages.chat_message("user").write(store_messages_user[i])
        messages.chat_message("assistant").write(store_messages_bot[i])

if prompt := st.chat_input("Say something"):
    for i in range(len(store_messages_user)):
        messages.chat_message("user").write(store_messages_user[i])
        messages.chat_message("assistant").write(store_messages_bot[i])

    messages.chat_message("user").write(prompt)
    store_messages_user.append(prompt)
    ans=get_answer(prompt)
    store_messages_bot.append(f"Bot: {ans}")
    st.session_state.store_messages_user = store_messages_user
    st.session_state.store_messages_bot = store_messages_bot
    messages.chat_message("assistant").write(f"Bot: {ans}")