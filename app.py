from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the movie data
with open('model/movie_predictor.pickle', 'rb') as file:
    movies_data = pickle.load(file)

# Assuming you want to use cosine similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_data['overview'].fillna(''))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(movie_title):
    idx = movies_data.index[movies_data['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get the top 5 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies_data['title'].iloc[movie_indices].tolist()

@app.route('/', methods=['POST', 'GET'])
def index():
    recommendations = []

    if request.method == 'POST':
        movie_title = request.form['movie_title']

        if movie_title in movies_data['title'].values:
            recommendations = get_recommendations(movie_title)

    return render_template("index_movies.html", recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
