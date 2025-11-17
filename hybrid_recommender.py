
import pickle
import streamlit as st
import requests
import os
import numpy as np
import pandas as pd

# === CONFIGURATION ===
# Hugging Face URLs
HF_SIMILARITY_URL = "https://huggingface.co/datasets/deep9234/movie_recommendation_files/resolve/main/similarity.pkl"
HF_MOVIE_URL = "https://huggingface.co/datasets/deep9234/movie_recommendation_files/resolve/main/movies.pkl"
HF_SVD_URL = "https://huggingface.co/datasets/deep9234/sad_model.pkl/resolve/main/svd_model.pkl" 
# This is the URL you provided
HF_LINKS_URL = "https://huggingface.co/datasets/deep9234/sad_model.pkl/resolve/main/links.csv"

FILES = {
    "movies.pkl": HF_MOVIE_URL,
    "similarity.pkl": HF_SIMILARITY_URL,
    "svd_model.pkl": HF_SVD_URL,
    "links.csv": HF_LINKS_URL
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
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Use 'wb' for binary files, 'w' for text
        mode = 'w' if destination.endswith('.csv') else 'wb'
        encoding = 'utf-8' if destination.endswith('.csv') else None
        
        with open(destination, mode, encoding=encoding) as f:
            if mode == 'w':
                # Write text content for CSV
                f.write(response.text)
            else:
                # Write binary chunks for pickle files
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
        st.success(f"✅ {destination} downloaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to download {destination}: {e}")
        st.stop()

def fetch_poster(movie_id):
    """Fetches movie poster from TMDB API."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=d9a5e1812b867e932d0efd41c90dd95a&language=en-US"
    placeholder_image = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found"
    try:
        data = requests.get(url, timeout=3) # Increased timeout
        data.raise_for_status()
        data = data.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return placeholder_image
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except requests.exceptions.RequestException:
        return placeholder_image

# === INITIALIZATION ===
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

# Download all necessary files
for filename, url in FILES.items():
    download_file_v2(url, filename)

# Load Data and Create ID Mapper
try:
    with open("movies.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
        
    # Load the links file to map TMDB IDs to MovieLens IDs
    links_df = pd.read_csv("links.csv")
    links_df['tmdbId'] = pd.to_numeric(links_df['tmdbId'], errors='coerce')
    links_df['movieId'] = pd.to_numeric(links_df['movieId'], errors='coerce')
    links_df = links_df.dropna(subset=['tmdbId', 'movieId'])
    
    # Create the mapping dictionary: {tmdbId: movieId}
    # We must cast tmdbId to int for the lookup to work
    tmdb_to_movielens_map = pd.Series(links_df.movieId.values, index=links_df.tmdbId.astype(int)).to_dict()

except Exception as e:
    st.error(f"❌ Critical Error loading data files: {e}")
    st.stop()

# === COLLABORATIVE FILTERING SETUP (SVD) ===
svd_model = None
try:
    if os.path.exists("svd_model.pkl"):
        with open("svd_model.pkl", "rb") as f:
            svd_model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load SVD model: {e}")
    pass # App will continue with mock logic

def get_collaborative_score(user_id, tmdb_id, model=None, id_map=None):
    """
    Predicts a rating (1-5) for a user and a movie.
    Uses the id_map to convert TMDB ID to MovieLens ID.
    """
    if model and id_map:
        # 1. Convert TMDB ID (which is a float in 'movies' df) to int for lookup
        tmdb_id_int = int(tmdb_id)
        
        # 2. Find the corresponding MovieLens ID
        movielens_id = id_map.get(tmdb_id_int)
        
        if movielens_id is None:
            # This movie wasn't in the MovieLens dataset, return neutral score
            return 3.0 

        try:
            # 3. Predict using the MovieLens ID
            pred = model.predict(uid=user_id, iid=movielens_id).est
            return pred
        except Exception:
            # Model prediction failed (e.g., movie not in trainset)
            return 3.0 # Neutral fallback
    else:
        # === MOCK LOGIC (IF SVD FAILED TO LOAD) ===
        import random
        return random.uniform(2.5, 5.0)

# === HYBRID ENGINE ===
def hybrid_recommend(movie_name, alpha=0.6):
    """
    Hybrid Recommendation Logic.
    """
    try:
        # Find the index of the selected movie
        movie_idx = movies[movies["title"] == movie_name].index[0]
    except IndexError:
        st.error("Movie not found in the database.")
        return [], [], []

    # Get similarity scores (list of (index, score))
    sim_scores = list(enumerate(similarity[movie_idx]))
    
    # Sort by similarity and take top 30 candidates
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:31]
    
    hybrid_candidates = []
    current_user_id = 1 # Fictitious user ID for demo
    
    for idx, content_score in sim_scores:
        movie_data = movies.iloc[idx]
        title = movie_data.title
        tmdb_id = movie_data.movie_id # This is the TMDB ID
        
        # Get predicted rating from SVD (Scale 1-5)
        svd_rating = get_collaborative_score(current_user_id, tmdb_id, svd_model, tmdb_to_movielens_map)
        
        # Normalize SVD rating to 0-1 scale to match cosine similarity
        normalized_svd = (svd_rating - 0.5) / 4.5 # More accurate normalization (0.5 to 5 scale)
        
        # Calculate Hybrid Score
        final_score = (content_score * alpha) + (normalized_svd * (1 - alpha))
        
        hybrid_candidates.append((title, tmdb_id, final_score, svd_rating))

    # Sort by Final Hybrid Score
    hybrid_candidates = sorted(hybrid_candidates, key=lambda x: x[2], reverse=True)
    
    # Get Top 5
    top_5 = hybrid_candidates[:5]
    
    names = [x[0] for x in top_5]
    posters = [fetch_poster(x[1]) for x in top_5]
    debug_ratings = [x[3] for x in top_5]
    
    return names, posters, debug_ratings

# === UI DESIGN ===
st.title("🎬 Hybrid Movie Recommender")
st.markdown("Combines **Content-Based Filtering** (Movie Similarity) and **Collaborative Filtering** (Predicted User Ratings).")

# Sidebar Controls
st.sidebar.header("⚙️ Hybrid Settings")
hybrid_weight = st.sidebar.slider(
    "Hybrid Weight (Alpha)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7, 
    step=0.1,
    help="1.0 = Pure Content Similarity. 0.0 = Pure User Rating Prediction."
)

if svd_model is None:
    st.sidebar.warning("⚠️ SVD Model not found. Using simulated ratings for demo.")
else:
    st.sidebar.success("✅ SVD Model Active")

# Main Interface
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
                # Fixed the deprecated parameter
                st.image(poster, use_container_width=True) 
                st.markdown(f"**{name}**")
                st.caption(f"Predicted Rating: ⭐ {rating:.1f}/5")

except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
    st.exception(e) # Show full error details

# === EXPLANATION SECTION ===
with st.expander("ℹ️ How this Hybrid System works"):
    st.write("""
    1. **Content Filter:** Finds the top 30 movies similar to your selection (using `similarity.pkl`).
    2. **ID Mapping:** Uses `links.csv` to convert the TMDB ID for each movie into its matching MovieLens ID.
    3. **Collaborative Filter:** Predicts how a demo user would rate those 30 movies (using `svd_model.pkl` and the MovieLens ID).
    4. **Hybrid Engine:** Combines these two scores using the formula:
       $$ Score = (Similarity \\times \\alpha) + (\\frac{PredictedRating}{5} \\times (1 - \\alpha)) $$
    5. **Result:** Returns the top 5 movies based on this final hybrid score.
    """)















