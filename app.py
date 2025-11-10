import pickle
import streamlit as st
import requests
import os
def download_from_gdrive(file_id, destination):
    """Download large files from Google Drive without gdown"""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

similarity_file_id = "1Pgp0-bt5NJkj3LzkqOU2QkoOPagujymK"

if not os.path.exists("similarity.pkl"):
    with st.spinner("Downloading similarity matrix..."):
        download_from_gdrive(similarity_file_id, "similarity.pkl")
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d9a5e1812b867e932d0efd41c90dd95a&language=en-US"
    placeholder_image = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"
    
    try:
        data = requests.get(url)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return placeholder_image
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException:
        return placeholder_image

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movie_names = []
    recommended_movie_posters = []
    
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    
    return recommended_movie_names, recommended_movie_posters

st.header('🎬 Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    names, posters = recommend(selected_movie)
    cols = st.columns(5)
    for col, name, poster in zip(cols, names, posters):
        with col:
            st.text(name)
            st.image(poster)














