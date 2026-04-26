import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies.pkl
movies = pickle.load(open('movies.pkl', 'rb'))

# Load similarity.pkl if exists, otherwise generate it
try:
    Cos = pickle.load(open('similarity.pkl', 'rb'))
except:
    cv = CountVectorizer(stop_words='english', max_features=5000)
    vectors = cv.fit_transform(movies['tags']).toarray()
    Cos = cosine_similarity(vectors)
    # Save for next time
    pickle.dump(Cos, open('similarity.pkl', 'wb'))

# Recommend function
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = Cos[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended = []
    for i in movie_list:
        recommended.append(movies.iloc[i[0]].title)
    return recommended

# UI
st.title("🎬 Movie Recommender System")
st.markdown("Select a movie and get 5 similar movie recommendations!")

selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    results = recommend(selected_movie)
    st.subheader("Top 5 Recommendations:")
    for i, movie in enumerate(results, 1):
        st.write(f"**{i}.** {movie}")