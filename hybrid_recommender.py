import pickle
import streamlit as st
import requests
import os
import numpy as np
import pandas as pd

# === CONFIGURATION ===
# Corrected Hugging Face URLs
HF_BASE = "https://huggingface.co/datasets/deep9234/movie_recommendation_files/resolve/main/"
FILES = {
    "movies.pkl": HF_BASE + "movies.pkl",
    "similarity.pkl": HF_BASE + "similarity.pkl",
    "svd_model.pkl": HF_BASE + "svd_model.pkl",
    "links.csv": HF_BASE + "links.csv",
}

# === HELPER FUNCTIONS ===
def download_file_v2(url, destination):
    """
    Downloads a file from a URL. Handles both text (.csv) and binary (.pkl) files.
    Skips if file already exists.
    """
    if os.path.exists(destination):
        return
    st.write(f"📥 Downloading {destination} ...")
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        mode = 'w' if destination.endswith('.csv') else 'wb'
        encoding = 'utf-8' if destination.endswith('.csv') else None
        
        with open(destination, mode, encoding=encoding) as f:
            if mode == 'w':
                f.write(response.text)
            else:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)

        # ✅ Validate file size
        if os.path.getsize(destination) < 1024:  # less than 1 KB likely corrupted
            raise ValueError(f"Downloaded file {destination} looks empty or invalid.")
        
        st.success(f"✅ {destination} downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download {destination}: {e}")
        if os.path.exists(destination):
            os.remove(destination)  # remove partial file
        st.stop()

def safe_load_pickle(path):
    """Safely loads a pickle file and retries if corrupted."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Failed to load {path}: {e}")
        if os.path.exists(path):
            os.remove(path)
        st.info(f"🔄 Re-downloading {path} ...")
        download_file_v2(FILES[path], path)
        with open(path, "rb") as f:
            return pickle.load(f)

def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d9a5e1812b867e932d0efd41c90dd95a&language=en-US"
    placeholder_image = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"
    try:
        data = requests.get(url, timeout=3)
        data.raise_for_status()
        data = data.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return placeholder_image
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException:
        return placeholder_image

# === INITIALIZATION ===
st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="wide")

# Download all necessary files
for filename, url in FILES.items():
    download_file_v2(url, filename)

# === LOAD DATA ===
try:
    movies = safe_load_pickle("movies.pkl")
    similarity = safe_load_pickle("similarity.pkl")

    # Load and clean links.csv
    links_df = pd.read_csv("links.csv")
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    links_df['movieId'] = pd.to_numeric(links_df['movieId'], errors='coerce')
    links_df = links_df.dropna(subset=['tmdbId', 'movieId'])
    tmdb_to_movielens_map = pd.Series(links_df.movieId.values, index=links_df.tmdbId.astype(int)).to_dict()

except Exception as e:
    st.error(f"❌ Critical Error loading data files: {e}")
    st.stop()

# === LOAD SVD MODEL ===
svd_model = None
try:
    if os.path.exists("svd_model.pkl"):
        with open("svd_model.pkl", "rb") as f:
            svd_model = pickle.load(f)
except Exception as e:
    st.warning(f"⚠️ Failed to load SVD model: {e}")
    svd_model = None

# === COLLABORATIVE FILTERING ===
def get_collaborative_score(user_id, tmdb_id, model=None, id_map=None):
    if model and id_map:
        tmdb_id_int = int(tmdb_id)
        movielens_id = id_map.get(tmdb_id_int)
        if movielens_id is None:
            return 3.0
        try:
            pred = model.predict(uid=user_id, iid=movielens_id).est
            return pred
        except Exception:
            return 3.0
    else:
        import random
        return random.uniform(2.5, 5.0)

# === HYBRID RECOMMENDER ===
def hybrid_recommend(movie_name, alpha=0.6):
    try:
        movie_idx = movies[movies["title"] == movie_name].index[0]
    except IndexError:
        st.error("Movie not found in the database.")
        return [], [], []

    sim_scores = list(enumerate(similarity[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:31]

    hybrid_candidates = []
    current_user_id = 1

    for idx, content_score in sim_scores:
        movie_data = movies.iloc[idx]
        title = movie_data.title
        tmdb_id = movie_data.movie_id
        
        svd_rating = get_collaborative_score(current_user_id, tmdb_id, svd_model, tmdb_to_movielens_map)
        normalized_svd = (svd_rating - 0.5) / 4.5
        final_score = (content_score * alpha) + (normalized_svd * (1 - alpha))
        hybrid_candidates.append((title, tmdb_id, final_score, svd_rating))

    hybrid_candidates = sorted(hybrid_candidates, key=lambda x: x[2], reverse=True)
    top_5 = hybrid_candidates[:5]
    names = [x[0] for x in top_5]
    posters = [fetch_poster(x[1]) for x in top_5]
    debug_ratings = [x[3] for x in top_5]
    return names, posters, debug_ratings

# === UI ===
st.title("🎬 Hybrid Movie Recommender")
st.markdown("Combines **Content-Based Filtering** and **Collaborative Filtering (SVD)** to find similar movies.")

st.sidebar.header("⚙️ Hybrid Settings")
hybrid_weight = st.sidebar.slider(
    "Hybrid Weight (Alpha)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="1.0 = Pure Content Similarity, 0.0 = Pure Collaborative Rating."
)

if svd_model is None:
    st.sidebar.warning("⚠️ SVD Model not found. Using simulated ratings for demo.")
else:
    st.sidebar.success("✅ SVD Model Active")

try:
    movie_list = movies["title"].values
    selected_movie = st.selectbox("🎥 Select a movie you like:", movie_list)

    if st.button("🔍 Get Recommendations"):
        with st.spinner("Calculating Hybrid Scores..."):
            names, posters, ratings = hybrid_recommend(selected_movie, alpha=hybrid_weight)
        st.subheader(f"Because you liked '{selected_movie}':")
        cols = st.columns(5)
        for col, name, poster, rating in zip(cols, names, posters, ratings):
            with col:
                st.image(poster, use_container_width=True)
                st.markdown(f"**{name}**")
                st.caption(f"Predicted Rating: ⭐ {rating:.1f}/5")

except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
    st.exception(e)

with st.expander("ℹ️ How this Hybrid System works"):
    st.write("""
    1. **Content Filter:** Finds the top 30 movies similar to your selection (using `similarity.pkl`).
    2. **ID Mapping:** Converts TMDB IDs to MovieLens IDs via `links.csv`.
    3. **Collaborative Filter:** Predicts ratings using the SVD model (`svd_model.pkl`).
    4. **Hybrid Engine:** Combines both using:
       $$ Score = (Similarity \\times \\alpha) + (\\frac{PredictedRating}{5} \\times (1 - \\alpha)) $$
    5. **Output:** Top 5 hybrid recommendations with posters and predicted ratings.
    """)















