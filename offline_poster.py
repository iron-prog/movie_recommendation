import pickle, requests, pandas as pd, time

API_KEY = ''
movie = pickle.load(open('/Users/deepaktiwari/Downloads/movies.pkl', 'rb'))

def fetch_poster_url(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
        data = requests.get(url, timeout=10).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path'] if data.get('poster_path') else None
    except:
        return None

movie['poster_url'] = movie['movie_id'].apply(fetch_poster_url)
movie.to_pickle('movies_with_posters.pkl')
