from sentence_transformers import SentenceTransformer, util
import kagglehub
import pandas as pd
import numpy as np
# Download latest version
path = kagglehub.dataset_download("huhuyngun/vietnamese-chatbot-ver2")
# doc du lieu 
qa_dataset=pd.read_csv(path+"/vi-QA.csv")
# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')
# encode the questions in the dataset
question_embeddings= model.encode(qa_dataset['questions'])
# user question
user_question = "Thích phim gì?"
user_embedding = model.encode(user_question)
# Calculate cosine similarity

scores = util.cos_sim(user_embedding, question_embeddings)
# find the index of the best match
best_idx = np.argmax(scores)
# Print the answer corresponding to the best match
print("Trả lời:", qa_dataset['answers'][best_idx])