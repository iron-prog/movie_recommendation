import pickle
import streamlit as st
import requests
import pickle, gdown, os

file_id = "1Pgp0-bt5NJkj3LzkqOU2QkoOPagujymK"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("similarity.pkl"):
    gdown.download(url, "similarity.pkl", quiet=False)

similarity = pickle.load(open("similarity.pkl", "rb"))


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=d9a5e1812b867e932d0efd41c90dd95a&language=en-US".format(movie_id)
    # Placeholder image in case the API fails
    placeholder_image = "https://via.placeholder.com/500x750.png?text=Poster+Not+Found" 
    
    try:
        data = requests.get(url)
        data.raise_for_status() # This will raise an error for bad responses (404, 500, etc.)
        data = data.json()
        
        poster_path = data.get('poster_path') # Use .get() to avoid errors if 'poster_path' is missing
        
        if not poster_path:
            return placeholder_image
            
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path

    except requests.exceptions.RequestException as e:
        # This will catch ConnectionError, Timeout, HTTPError, etc.
        st.error(f"API Error: Could not fetch poster for movie ID {movie_id}. Details: {e}")
        return placeholder_image

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters


st.header('Movie Recommender System')
movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    # This is the corrected line:
    col1, col2, col3, col4, col5 = st.columns(5) 
    
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])













