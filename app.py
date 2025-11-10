import pickle
import streamlit as st
import requests
import os

# === CONFIG ===
# Your Google Drive file ID (make sure file is set to "Anyone with link → Viewer")
FILE_ID = "1Pgp0-bt5NJkj3LzkqOU2QkoOPagujymK"
MOVIE_FILE = "movies.pkl"
SIMILARITY_FILE = "similarity.pkl"


# === DOWNLOAD FUNCTION ===
def download_from_gdrive(file_id, destination):
    """Downloads a file from Google Drive, handling confirmation tokens and large files."""
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    # Handle confirmation token (for large files)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    # Verify file is a real pickle (not HTML)
    with open(destination, "rb") as f:
        start = f.read(20)
        if b"<html" in start.lower():
            os.remove(destination)
            st.error("❌ Google Drive returned an HTML file (not a pickle). Check that your file is public and under the download limit.")
            st.stop()


# === LOAD OR DOWNLOAD FILE ===
if not os.path.exists(SIMILARITY_FILE):
    with st.spinner("📥 Downloading similarity.pkl from Google Drive..."):
        download_from_gdrive(FILE_ID, SIMILARITY_FILE)

# === LOAD PICKLE ===
try:
    similarity = pickle.load(open(SIMILARITY_FILE, 'rb'))
except Exception as e:
    st.error(f"❌ Failed to unpickle similarity.pkl: {e}")
    st.stop()


# === LOAD MOVIE DATA ===
try:
    movies = pickle.load(open(MOVIE_FILE, 'rb'))
except FileNotFoundError:
    st.error("❌ movies.pkl not found. Please upload or download it.")
    st.stop()


# === POSTER FETCH ===
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


# === RECOMMENDER ===
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


# === STREAMLIT UI ===
st.header('🎬 Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "🎥 Type or select a movie from the dropdown",
    movie_list
)

if st.button('🔍 Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)
    for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
        with col:
            st.text(name)
            st.image(poster)















