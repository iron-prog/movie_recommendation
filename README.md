🎬 Hybrid Movie Recommender System

A Streamlit web app that provides hybrid movie recommendations, blending Content-Based Filtering (what the movie is about) with Collaborative Filtering (what users think).

🛑 ACTION REQUIRED 🛑

<img width="1440" height="900" alt="Screenshot 2025-11-17 at 7 36 14 PM" src="https://github.com/user-attachments/assets/08a3eb5d-49b3-4177-98bc-3452c0ae6bc9" />


<img width="1440" height="900" alt="Screenshot 2025-11-17 at 7 36 38 PM" src="https://github.com/user-attachments/assets/3b4ac7c4-66c9-4f6f-83b7-b78e48a24b61" />

``

▶️ View Live Demo

https://movierecommendation-rhthwzjde93mwersqsqg93.streamlit.app

🌟 Core Features

Hybrid Recommendation Engine: Combines two different recommendation strategies for more robust and personalized results.

Content-Based Filtering: Uses a pre-computed similarity.pkl (Cosine Similarity) to find movies with similar metadata (genres, keywords, cast, etc.).

Collaborative Filtering: Employs a trained SVD (Singular Value Decomposition) model (svd_model.pkl) to predict a user's rating for a movie, based on historical ratings from the MovieLens dataset.

Interactive UI: Built with Streamlit, the app features an interactive slider to adjust the weight (alpha) between the two recommendation styles.

Dynamic Content: Fetches movie posters and details in real-time from the TMDB API.

Cloud-Hosted Models: All data and model files (.pkl, .csv) are downloaded on the fly from a Hugging Face Dataset repository, making the app lightweight.

⚙️ How It Works

The hybrid logic provides a powerful "best of both worlds" approach:

Select a Movie: The user picks a movie they like (e.g., "Spider Man 3").

Content-Based Candidates: The app finds the Top 30 movies that are most similar in content to "Spider Man 3" using the similarity.pkl matrix.

Collaborative Re-Ranking: The app then iterates through those 30 candidates and uses the svd_model.pkl to predict a rating for each one (e.g., "A demo user would rate Interstellar 4.8/5").

Calculate Hybrid Score: A final score is computed for each of the 30 movies using a weighted average, controlled by the "Alpha" slider in the UI:

$$\text{Score} = (\text{Similarity} \times \alpha) + (\text{Normalized\_SVD\_Rating} \times (1 - \alpha))$$

Final Recommendation: The app displays the Top 5 movies based on this final hybrid score, complete with their posters and predicted ratings.

🚀 How to Run Locally

1. Clone the Repository

git clone 
cd your-repo-name


2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate


3. Install Dependencies

The requirements.txt file contains all necessary libraries, including the specific numpy version to prevent compatibility issues.

pip install -r requirements.txt


4. Run the Streamlit App

Once dependencies are installed, you can run the app:

streamlit run hybrid_recommender.py


The app will open automatically in your browser. On first launch, it will download the model and data files from Hugging Face, which may take a moment.

🗂️ Project File Structure

.
├── hybrid_recommender.py   
├── requirements.txt        
├── train_svd.py           
├── ratings.csv            
└── README.md          


Note: The model files (movies.pkl, similarity.pkl, svd_model.pkl, links.csv) are not in the repo, as they are downloaded at runtime from the cloud.

📊 Data & Model Sources

Movie Data (Content): All movie metadata (genres, cast, keywords) and poster links are sourced from The Movie Database (TMDB).

Rating Data (Collaborative): The SVD model was trained on the MovieLens "Latest Small" Dataset, which contains 100,000 ratings.

Model & Data Hosting: All large data files and the trained SVD model are hosted on Hugging Face Datasets.

Content Data: deep9234/movie_recommendation_files

SVD Model: deep9234/sad_model.pkl

Built by Your Name. D.T

