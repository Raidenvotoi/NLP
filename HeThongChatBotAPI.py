import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

st.title("Hệ thống chatbot tạo sinh")

def get_answer(user):
    api_key = os.getenv("AIML_API_KEY")
    response = requests.post(
        "https://api.aimlapi.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": f"{user}"
                }
            ]
        }
    )

    data = response.json()
    return data['choices'][0]['message']['content']






if "store_messages_user" not in st.session_state:
    st.session_state.store_messages_user = []
    st.session_state.store_messages_bot = []
    
else :
    store_messages_user = st.session_state.get("store_messages_user", [])
    store_messages_bot = st.session_state.get("store_messages_bot", [])


messages = st.container(height=400)

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