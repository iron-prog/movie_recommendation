# Movie Recommendation 🎬

A simple movie recommendation web app built with Python and Flask.  
Given a movie selected by the user, the app suggests similar movies and shows their posters.

---

## 🚀 Features

- Select a movie from a drop-down list and get a list of recommended movies.  
- Displays movie posters alongside titles for each recommendation.  
- Works offline (uses a pre-processed pickle file of movies and similarity matrix).  
- Easy to deploy (Heroku / local) using `app.py`.

---

## 🧰 Technology Stack

- Python 3.x  
- Flask (for the web interface)  
- pandas / numpy (for data handling)  
- pickle (to store processed movie data)  
- A pre-computed similarity matrix (Cosine similarity or other)  
- `requirements.txt` lists all dependencies  
- `setup.sh` & `Procfile` included for easy deployment

---

## 📁 Project Structure

