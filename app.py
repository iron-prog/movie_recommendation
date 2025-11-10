import pickle
import streamlit as st
import requests
import os

# === CONFIG ===
# Hugging Face dataset hosting both movies.pkl and similarity.pkl
HF_SIMILARITY_URL = "https://huggingface.co/datasets/deep9234/movie_recommendation_files/resolve/main/similarity.pkl"
HF_MOVIE_URL = "https://huggingface.co/datasets/deep9234/movie_recommendation_files/resolve/main/movies.pkl"

MOVIE_FILE = "movies.pkl"
SIMILARITY_FILE = "similarity.pkl"


# === DOWNLOAD FUNCTION ===
def download_file(url, destination):
    """Download a binary file from Hugging Face with streaming."""
    st.write(f"📥 Downloading {destination} ...")
    response = requests.get(url, stream=True)
    try:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        st.success(f"✅ {destination} downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download {destination}: {e}")
        st.stop()


# === LOAD OR DOWNLOAD similarity.pkl ===
if not os.path.exists(SIMILARITY_FILE):
    with st.spinner("📥 Downloading similarity.pkl from Hugging Face..."):
        download_file(HF_SIMILARITY_URL, SIMILARITY_FILE)

# === LOAD OR DOWNLOAD movies.pkl ===
if not os.path.exists(MOVIE_FILE):
    with st.spinner("📥 Downloading movies.pkl from Hugging Face..."):
        download_file(HF_MOVIE_URL, MOVIE_FILE)

# === LOAD PICKLE FILES ===
try:
    with open(SIMILARITY_FILE, "rb") as f:
        similarity = pickle.load(f)
    st.success("✅ similarity.pkl loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load similarity.pkl: {e}")
    st.stop()

try:
    with open(MOVIE_FILE, "rb") as f:
        movies = pickle.load(f)
    st.success("✅ movies.pkl loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load movies.pkl: {e}")
    st.stop()


# === POSTER FETCH FUNCTION ===
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d9a5e1812b867e932d0efd41c90dd95a&language=en-US"
    placeholder_image = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"
    try:
        data = requests.get(url)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return placeholder_image
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException:
        return placeholder_image


# === RECOMMENDER FUNCTION ===
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters


# === STREAMLIT UI ===
st.header("🎬 Movie Recommender System")

try:
    movie_list = movies["title"].values
    selected_movie = st.selectbox("🎥 Type or select a movie from the dropdown", movie_list)

    if st.button("🔍 Show Recommendation"):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        cols = st.columns(5)
        for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
            with col:
                st.text(name)
                st.image(poster)
except Exception as e:
    st.error(f"⚠️ Something went wrong: {e}")
















